import torch
from torch import nn
from torch.nn import functional as F
# from skimage import color
from kornia import color
from detectron2.structures import ImageList
from ..utils import viz


def compute_project_term(mask_scores, gt_bitmasks, mode='max'):
    if mode == 'max':
        mask_losses_y = dice_coefficient(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        mask_losses_x = dice_coefficient(
            mask_scores.max(dim=3, keepdim=True)[0],
            gt_bitmasks.max(dim=3, keepdim=True)[0]
        )
    elif mode == 'avg':
        mask_losses_y = dice_coefficient(
            mask_scores.mean(dim=2, keepdim=True),
            gt_bitmasks.mean(dim=2, keepdim=True)
        )
        mask_losses_x = dice_coefficient(
            mask_scores.mean(dim=3, keepdim=True),
            gt_bitmasks.mean(dim=3, keepdim=True)
        )
    else:
        raise NotImplementedError
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

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


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(x, kernel_size=kernel_size, padding=padding, dilation=dilation)
    unfolded_x = unfolded_x.reshape(x.size(0), x.size(1), -1, x.size(2), x.size(3))

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(images, kernel_size=kernel_size, dilation=dilation)

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights


class BoxInstLoss(nn.Module):
    def __init__(self, cfg):
        super(BoxInstLoss, self).__init__()
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY

        self.mask_out_stride = 4

        self.register_buffer("_iter", torch.zeros([1]))

    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w):
        device = images.device
        stride = self.mask_out_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(images.float(), kernel_size=stride, stride=stride, padding=0)
        image_masks = image_masks[:, start::stride, start::stride]

        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = color.rgb_to_lab(downsampled_images[im_i] / 255.).unsqueeze(0)
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i], self.pairwise_size, self.pairwise_dilation
            )
            per_im_boxes = per_im_gt_inst.gt_boxes.tensor

            # vectorization
            num_ins = per_im_boxes.size()[0]
            per_im_boxes = per_im_boxes.int()
            bitmasks_full = torch.zeros((num_ins, im_h, im_w), device=device)
            for idx, per_box in enumerate(per_im_boxes):
                x1, y1, x2, y2 = per_box
                bitmasks_full[idx, y1:y2 + 1, x1:x2 + 1] = 1.0
            bitmasks = bitmasks_full[:, start::stride, start::stride]
            per_im_gt_inst.gt_bitmasks_full = bitmasks_full
            per_im_gt_inst.gt_bitmasks = bitmasks
            per_im_gt_inst.image_color_similarity = images_color_similarity.repeat((num_ins, 1, 1, 1))
            del bitmasks_full
            del bitmasks
            del images_color_similarity
            torch.cuda.empty_cache()
            # viz_boxes(images[0], per_im_gt_inst.gt_boxes)
            # visualize_color_similarity(per_im_gt_inst.image_color_similarity[0])
            # viz_pesudo_masks(per_im_gt_inst.gt_bitmasks)
            # print('s')

    def compute_pseudo_masks(self, batched_inputs):
        original_images = [x["image"] for x in batched_inputs]
        gt_instances = [x["instances"] for x in batched_inputs]

        original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images]

        # mask out the bottom area where the COCO dataset probably has wrong annotations
        for i in range(len(original_image_masks)):
            im_h = batched_inputs[i]["height"]
            pixels_removed = int(
                self.bottom_pixels_removed * float(original_images[i].size(1)) / float(im_h)
            )
            if pixels_removed > 0:
                original_image_masks[i][-pixels_removed:, :] = 0

        original_images = ImageList.from_tensors(original_images, self.size_divisibility)
        original_image_masks = ImageList.from_tensors(
            original_image_masks, self.size_divisibility, pad_value=0.0
        )
        self.add_bitmasks_from_boxes(
            gt_instances, original_images.tensor, original_image_masks.tensor,
            original_images.tensor.size(-2), original_images.tensor.size(-1)
        )

    def forward(self, mask_logits, gt_bitmasks, image_color_similarity):
        self._iter += 1
        device = mask_logits.device
        mask_logits = mask_logits.unsqueeze(1)

        if not len(mask_logits):
            dummy_loss = mask_logits.sum() * 0.
            return {"loss_prj"     : dummy_loss.item(),
                    # "loss_prj_avg" : dummy_loss.item(),
                    "loss_pairwise": dummy_loss.item()}

        mask_scores = mask_logits.sigmoid()

        loss_prj_term_max = compute_project_term(mask_scores, gt_bitmasks, mode='max')
        # loss_prj_term_avg = compute_project_term(mask_scores, gt_bitmasks, mode='avg')
        pairwise_losses = compute_pairwise_term(mask_logits, self.pairwise_size, self.pairwise_dilation)

        weights = (image_color_similarity.to(device) >= self.pairwise_color_thresh).float() * gt_bitmasks.float()

        # import matplotlib.pyplot as plt
        # for i in range(weights.shape[0]):
        #     plt.imshow(weights[i, 0, :, :].detach().cpu().numpy().copy())
        #     plt.show()

        loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

        warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
        loss_pairwise = loss_pairwise * warmup_factor

        return {"loss_prj"     : loss_prj_term_max,
                # "loss_prj_avg" : loss_prj_term_avg,
                "loss_pairwise": loss_pairwise}
