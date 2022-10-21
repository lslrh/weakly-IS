import torch
from torch.nn import functional as F
from torch import nn
import torch.distributed as dist

from fvcore.nn import sigmoid_focal_loss_jit
from adet.utils.comm import compute_locations, aligned_bilinear
from adet.utils.contrast import l2_normalize, momentum_update
from adet.utils.show import show_feature_map, show_feature_map_v2, show_feature_map_v3, show_feature_map_v4, show_feature_map_heatmap, show_feature_map_heat1
from adet.utils.sinkhorn import distributed_sinkhorn
from adet.layers import conv_with_kaiming_uniform
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from torch_scatter import scatter_mean, scatter
from adet.utils.comm import reduce_sum, reduce_mean, compute_ious
import math
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from palettable.cartocolors.sequential import DarkMint_4, RedOr_3, BluYl_3, Emrld_2, Sunset_3
INF = 1e8

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4
    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target, weights=None):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    if weights is not None:
        weights = weights.reshape(n_inst, -1)
        x = x*weights
        target = target*weights
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    if weights is not None:
        return loss.mean()
    return loss


def mask_focal_loss(x, targets, weights=None, alpha: float = 0.25, gamma: float = 2):
    ce_loss = F.binary_cross_entropy_with_logits(x, targets, weight=weights, reduction="none")
    p_t = x * targets + (1 - x) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()/(weights.sum()+1)


def mask_focal_loss_v2(x, targets, weights, alpha: float = 0.25, gamma: float = 2):
    ce_loss = F.binary_cross_entropy_with_logits(x, targets, weight=weights, reduction="none")
    p_t = x * targets + (1 - x) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()/(weights.sum()*x.shape[1]+1)  



def get_feat_similarity(images, kernel_size, dilation):
    assert images.dim() == 4
    from adet.modeling.condinst.condinst import unfold_wo_center

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
 
    similarity = torch.exp(-torch.norm(diff, dim=1) * 1)

    return similarity


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS
        self.num_classes = cfg.MODEL.BASIS_MODULE.NUM_CLASSES
        self.sem_in_channels = cfg.MODEL.BOXINST.SEM_IN_CHANNELS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self.pseudo_thresh = cfg.MODEL.BOXINST.PSEUDO_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        self.num_proto = cfg.MODEL.BOXINST.NUM_PROTO 
        self.sem_loss_on = cfg.MODEL.BOXINST.SEMANTIC_LOSS_ON
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_proto, self.in_channels), requires_grad=False)

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        # self.register_buffer("_iter", torch.zeros([1]))
        self._iter = 0
        trunc_normal_(self.prototypes, std=0.02)

        if self.sem_loss_on:
            norm = cfg.MODEL.CONDINST.MASK_BRANCH.NORM
            conv_block = conv_with_kaiming_uniform(norm, activation=True)
            in_channels = self.sem_in_channels
            channels = 128
            self.seg_head = nn.Sequential(
                conv_block(in_channels, channels, kernel_size=3, stride=1),
                conv_block(channels, channels, kernel_size=3, stride=1)
            )
            self.logits = nn.Conv2d(channels, self.num_classes, kernel_size=1, stride=1)
            
            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)
        
            for param in self.seg_head.parameters():
                param.requires_grad = False
            for param in self.logits.parameters():
                param.requires_grad = False


    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits

    def prototype_learning(self, mask_feat, masks, mask_scores, gt_classes, pred_labels, im_inds):
        for b in range(len(mask_feat)):
            gt_classes_unique = torch.unique(gt_classes[b])

            protos = self.prototypes.data.clone()
            for k in gt_classes_unique:
                inds = (im_inds==b) & (pred_labels==k)

                pseudo = mask_scores[inds]
                init_q = masks[inds]
                c_q = mask_feat[b].unsqueeze(0).expand(len(init_q), -1,-1,-1)

                init_q = rearrange(init_q, 'n p h w -> (n h w) p')
                pseudo = rearrange(pseudo, 'n p h w -> (n p h w)')
           
                c_q = rearrange(c_q, 'n h w p -> (n h w) p')

                init_q = init_q[pseudo==1]
                c_q = c_q[pseudo==1]

                if init_q.shape[0] == 0:
                    continue
                q, indexs = distributed_sinkhorn(init_q)

                f = q.transpose(0, 1) @ c_q
                f = F.normalize(f, p=2, dim=-1)
                n = torch.sum(q, dim=0)
                new_value = momentum_update(old_value=protos[k, n!=0, :], new_value=f[n!=0, :], momentum=0.999, debug=False)
                protos[k, n!=0, :] = new_value
            self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)
        
        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)
    
    def activate(self):
        for param in self.seg_head.parameters():
            param.requires_grad = True
        for param in self.logits.parameters():
            param.requires_grad = True

    def max_pooling_forward(self, z, pooling, strides=(2, 2), padding=(0, 0)):
        """
        最大池化前向过程
        :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
        :param pooling: 池化大小(k1,k2)
        :param strides: 步长
        :param padding: 0填充
        :return:
        """
        H, W = z.shape
        # 零填充
        padding_z = np.lib.pad(z, ( (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

        # 输出的高度和宽度
        out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
        out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

        pool_z = np.zeros((out_h, out_w))

        for i in np.arange(out_h):
            for j in np.arange(out_w):
                pool_z[i, j] = np.max(padding_z[
                                                    strides[0] * i:strides[0] * i + pooling[0],
                                                    strides[1] * j:strides[1] * j + pooling[1]])
        return pool_z
        

    def __call__(self, features, mask_feats, mask_feats_ema, mask_feat_stride, pred_instances, pred_instances_ema, gt_instances=None, batched_inputs=None):
        if self.training:
            self._iter += 1
            gt_inds = pred_instances.gt_inds

            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.boxinst_enabled:
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_prj"] = dummy_loss
                    losses["loss_pairwise"] = dummy_loss
            else:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_scores = mask_logits.sigmoid()
                
                mask_logits_ema = self.mask_heads_forward_with_coords(
                    mask_feats_ema, mask_feat_stride, pred_instances_ema
                )
                mask_scores_ema = mask_logits_ema.sigmoid()
                # show_feature_map(mask_feats[0].detach(), 0)
                # show_feature_map(mask_feats[1].detach(), 1)

                if self.boxinst_enabled:
                    # compute feats similarity 
                    mask_feats_ema = F.interpolate(
                                                mask_feats_ema,
                                                scale_factor=2,
                                                mode="bilinear", align_corners=False)
                    mask_feats = F.interpolate(
                                                mask_feats,
                                                scale_factor=2,
                                                mode="bilinear", align_corners=False)

                    mask_feat_similarity = get_feat_similarity(mask_feats_ema, self.pairwise_size, self.pairwise_dilation)
                    mask_feat_similarity_list = []
                    for i in range(len(gt_instances)):
                        mask_feat_similarity_list.append(torch.stack([mask_feat_similarity[i] for _ in range(len(gt_instances[i]))], dim=0))
                    mask_feat_similarity = torch.cat([x for x in mask_feat_similarity_list])
                    mask_feat_similarity = mask_feat_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

                    pairwise_losses = compute_pairwise_term(
                        mask_logits, self.pairwise_size,
                        self.pairwise_dilation
                    )

                    image_similarity = mask_feat_similarity * image_color_similarity
                    # image_similarity = image_color_similarity
                    weights = (image_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

                    warmup_factor = min(self._iter / float(self._warmup_iters), 1.0)
                    loss_pairwise = loss_pairwise * warmup_factor

                    # compute pseudo loss
                    proto_masks_list = []
                    self.prototypes.data.copy_(l2_normalize(self.prototypes))
                    mask_feats_ema = l2_normalize(mask_feats_ema).permute(0,2,3,1)
                    mask_feats = l2_normalize(mask_feats).permute(0,2,3,1)

                    for i, x in enumerate(gt_instances):
                        masks = torch.einsum('hwd,npd->nphw', mask_feats_ema[i], self.prototypes[x.gt_classes]).detach()
                        proto_masks_list.append(masks)
                    proto_masks = torch.cat([x for x in proto_masks_list])
                    proto_masks = proto_masks[gt_inds]
                    
                    ious, _ = compute_ious(pred_instances.reg_pred, pred_instances.reg_targets)
                    ious = torch.exp(ious*10)
                    ious_sum = scatter(ious, gt_inds, dim=0, reduce="sum")[gt_inds]
                    ious_weight = ious/ious_sum
                    # pdb.set_trace()
                    (h, w) = batched_inputs[0]['image'].shape[1:]
                    heat_map = np.zeros((h, w)) 
                    locations = pred_instances.locations[pred_instances.im_inds==0].cpu().numpy()
                    for i, l in enumerate(locations):
                        heat_map[int(l[1])][int(l[0])] = ious_weight[pred_instances.im_inds==0][i] 

                    input_tensor =  batched_inputs[0]['image'].clone()
                    input_tensor = input_tensor.to(torch.device('cpu')).permute(1,2,0).numpy()
                    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
                    input_tensor = Image.fromarray(input_tensor)
                    input_tensor.save('feat_visualize/img1.png', 'png')

                    heat_map = heat_map/heat_map.max()
                    heat_map = self.max_pooling_forward(heat_map, (16,16))*255    
                    heat_map = cv2.resize(heat_map.astype(np.uint8), (w,h))
                    
                    img = cv2.cvtColor(np.asarray(input_tensor),cv2.COLOR_RGB2BGR)  

                    heat_img = cv2.applyColorMap(heat_map, cv2.COLORMAP_sumer)

                    add_img = cv2.addWeighted(img, 0.5, heat_img, 0.5, 0)
                    input_tensor = Image.fromarray(add_img)
                    input_tensor.save('feat_visualize/add_img.png')
                    # cv2.imwirte('feat_visualize/add_img.png', add_img)
                    pdb.set_trace()
                    
                    # plt.imshow(heat_map)
                    # plt.axis('off')
                    #     # scipy.misc.imsave(str(index)+".png", feature_map[index-1])
                    # plt.savefig(os.path.join('feat_visualize', "heatmap.png"))
                    
                    mask_scores_mean = scatter(mask_scores_ema*ious_weight[:,None,None,None], gt_inds.unsqueeze(1), dim=0, reduce="sum")[gt_inds]
                    # mask_scores_mean = scatter_mean(mask_scores_ema, gt_inds.unsqueeze(1), dim=0)[gt_inds]
                    # compute the threshold for pseudo labels
                    masks = (mask_scores_mean > 0.7) * gt_bitmasks.float()
                    if self._iter > 0:
                        n_pos = torch.sum(masks, dim=(2,3))
                        proto_seg = torch.amax(proto_masks, dim=1).unsqueeze(1).sigmoid()
                        pseudo_seg = 0.5 * proto_seg + 0.5 * mask_scores_mean
                        pseudo_scores = (pseudo_seg * gt_bitmasks).reshape(len(gt_inds), -1) 
                        pseudo_scores_rank = torch.sort(pseudo_scores, descending=True)[0]
                        n_pos = torch.clamp(n_pos, min=0, max=pseudo_scores_rank.shape[1]-1).long()
                      
                        assert n_pos.max() < pseudo_scores_rank.shape[1]
                        thr = torch.gather(pseudo_scores_rank, dim=1, index=n_pos)
                        thr = scatter_mean(thr.squeeze(1), gt_inds)[gt_inds][:,None,None,None]
                        thr = torch.clamp(thr, min=0.7, max=0.8)
                        pseudo_seg_final = ( pseudo_seg > thr) * gt_bitmasks.float()
                        
                        # show_feature_map(pseudo_seg_final.detach(), 2)
                        # show_feature_map((mask_scores_ema).detach(), 3)
                        # show_feature_map(((proto_seg>0.6)* gt_bitmasks.float()).detach(), 4)
                        # masks_2 = (mask_scores_mean > 0.65) * gt_bitmasks.float()
                        # show_feature_map(masks_2.detach(), 5)
                        # pdb.set_trace()
                    
                        warmup_factor_2 = min(self._iter / float(self.max_iter), 1)
                        weights = ((pseudo_seg > thr) | (pseudo_seg < 0.3)) * gt_bitmasks
                        loss_pseudo = (mask_focal_loss(mask_scores, pseudo_seg_final.detach(), weights)) * warmup_factor_2 * 0.3
                        losses["loss_pseudo"] = loss_pseudo

                        # compute the paste masks
                        # mask_scores_mean = scatter_mean(mask_scores_ema, gt_inds.unsqueeze(1), dim=0)
                        mask_scores_mean = scatter(mask_scores_ema*ious_weight[:,None,None,None], gt_inds.unsqueeze(1), dim=0, reduce="sum")                      
                        mask_scores_mean = F.interpolate(
                                                        mask_scores_mean,
                                                        scale_factor=4,
                                                        mode="bilinear", align_corners=False)

                        paste_mask = mask_scores_mean > 0.7
                        # paste_weight = (mask_scores_mean > 0.7) | (mask_scores_mean < 0.3)
                        start = 0
                        gt_bitmasks_full = torch.cat([per_im.gt_bitmasks_full for per_im in gt_instances])
                        for per_im in gt_instances:
                            end = start + len(per_im)
                            if len(per_im)!= len(paste_mask[start:end]):
                                continue
                            per_im.paste_mask = paste_mask[start:end].squeeze(1) * gt_bitmasks_full[start:end]
                            # per_im.paste_weight = paste_weight[start:end].squeeze(1)
                            score = (mask_scores_mean[start:end].squeeze(1) * per_im.paste_mask).sum(dim=[1,2]) \
                                / (per_im.paste_mask.sum(dim=[1,2])+1)
                            per_im.score = score
                            start = end
                        # compute mask loss on pasted instances
                        if 'paste_indicator' in gt_instances[0]._fields.keys():
                            gt_masks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])[gt_inds].unsqueeze(1)
                            # weights = torch.cat([per_im.gt_weight for per_im in gt_instances])[gt_inds].unsqueeze(1)
                            _, _, H, W = gt_masks.shape
                            indicator = torch.ones(H, W)[None,None,:,:].repeat(len(gt_masks),1,1,1).to(mask_scores.device)
                            weights = torch.cat([per_im.paste_indicator for per_im in gt_instances])[gt_inds][:,None,None,None].to(mask_scores.device) * indicator
                            # show_feature_map(gt_masks.detach(), 4)
                            # show_feature_map(weights.detach(), 5)
                            # pdb.set_trace()
                            loss_paste = mask_focal_loss(mask_scores, gt_masks, weights) * 1.0 + dice_coefficient(mask_scores, gt_masks, weights)
                            losses['loss_paste'] = loss_paste
                        else:
                            losses['loss_paste'] = torch.tensor(0).to(mask_scores.device)

                        if self.sem_loss_on:
                            self.activate()
                            logits_pred = self.logits(self.seg_head(
                                                features['p3']
                                            ))
                            semantic_targets = []
                            pseudo_neg_list = []
                            for b in range(len(gt_instances)):
                                inds = pred_instances_ema.im_inds==b
                                h, w = pseudo_seg_final.size()[-2:]
                                pseudo_neg = ( pseudo_seg > 0.3) * gt_bitmasks.float()
                                if inds.sum()>0:
                                    pseudo = pseudo_seg_final.detach()[inds]
                                    gt_inds_b = gt_inds[inds]
                                    pseudo_neg = pseudo_neg.detach()[inds]
                                    mask= torch.ones_like(gt_inds_b)
                                    mask[1:] = gt_inds_b[1:] != gt_inds_b[:-1]
                                    pseudo = pseudo[mask.bool()].squeeze(1)
                                    pseudo_neg = pseudo_neg[mask.bool()].squeeze(1).sum(0)
                                    # show_feature_map(pseudo.unsqueeze(1), 6)
                                    areas = pseudo.sum(dim=-1).sum(dim=-1)
                                    areas = areas[:, None, None].repeat(1, h, w)
                                    areas[pseudo == 0] = INF
                                    areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                                    min_areas, inds = areas.min(dim=1)
                                
                                    per_im_sematic_targets = gt_instances[b].gt_classes[inds] + 1
                                    per_im_sematic_targets[min_areas == INF] = 0
                                    per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)
                                else: 
                                    per_im_sematic_targets = torch.zeros(h, w).to(logits_pred.device)
                                    pseudo_neg = torch.zeros(h, w).to(logits_pred.device)
                                semantic_targets.append(per_im_sematic_targets)
                                pseudo_neg_list.append(pseudo_neg)
                            
                            semantic_targets = torch.stack(semantic_targets, dim=0)
                            pseudo_neg = torch.stack(pseudo_neg_list, dim=0)
                            # resize target to reduce memory
                            semantic_targets = semantic_targets[:, None, 1::2, 1::2]
                            pseudo_neg = pseudo_neg[:, None, 1::2, 1::2]
                            weights_pos = semantic_targets > 0.5
                            weights_neg = ~(pseudo_neg > 0.5)
                            weights = weights_pos | weights_neg
                            # prepare one-hot targets
                            num_classes = logits_pred.size(1)
                            class_range = torch.arange(
                                num_classes, dtype=logits_pred.dtype,
                                device=logits_pred.device
                            )[:, None, None]
                            class_range = class_range + 1
                            one_hot = (semantic_targets == class_range).float()
                            loss_sem = mask_focal_loss_v2(
                                torch.sigmoid(logits_pred), one_hot, weights
                            ) * warmup_factor_2 * 0.1
                            losses['loss_sem'] = loss_sem
                            
                    # update the prototypes
                    gt_classes = [x.gt_classes for x in gt_instances]
                    self.prototype_learning(mask_feats, proto_masks, masks, gt_classes, pred_instances_ema.labels, pred_instances_ema.im_inds)

                    losses.update({
                        "loss_prj": loss_prj_term,
                        "loss_pairwise": loss_pairwise,
                    })
                else:
                    # fully-supervised CondInst losses
                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                    losses["loss_mask"] = loss_mask
            return losses

        else:
            if len(pred_instances) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances