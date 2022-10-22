# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
from multiprocessing import dummy
import cv2
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from mask2former.utils import box_ops
from mask2former.modeling.boxinst_loss import BoxInstLoss
import pdb

def flat_gt_ids(tgt_indices):
    new_tgt_indices = []
    offset = 0
    for ids in tgt_indices:
        new_tgt_indices.append(ids + offset)
        offset = len(ids)
    if len(new_tgt_indices) == 0:
        return torch.tensor(new_tgt_indices)
    return torch.cat(new_tgt_indices)


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, cfg):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            cfg: config for weakly supervised instances segmentation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # weakly instance segmentation
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.boxinst_loss = BoxInstLoss(cfg)

        self.register_buffer("_iter", torch.zeros([1]))
        self._warmup_iters = 10000

        # weak and strong augmentation setting
        self.weak_aug = cfg.INPUT.WEAK_AUG
        self.strong_aug = cfg.INPUT.STRONG_AUG
        self.strong_with_proj_loss = cfg.MODEL.BOXINST.STRONG_AUG.PROJ_LOSS
        self.strong_with_pair_loss = cfg.MODEL.BOXINST.STRONG_AUG.PAIR_LOSS
        self.consistency_cfg = {'loss_type' : cfg.MODEL.CONSISTENCY.LOSS_TYPE,
                                'label_type': cfg.MODEL.CONSISTENCY.LABEL_TYPE,
                                'mining'    : {
                                    'enabled'   : cfg.MODEL.CONSISTENCY.MINING.ENABLED,
                                    'min_thresh': cfg.MODEL.CONSISTENCY.MINING.MIN_THRESH,
                                    'max_thresh': cfg.MODEL.CONSISTENCY.MINING.MAX_THRESH}
                                }

    def loss_labels(self, outputs, targets, indices, num_masks, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][src_idx].float()
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # TODO: optimizing background without objects
        if len(src_idx[0]) == 0:
            return {'loss_bbox': torch.mean(outputs['pred_boxes']) * 0.,
                    'loss_giou': torch.mean(outputs['pred_boxes']) * 0.}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = dict()
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, **kwargs):
        """Compute the losses related to the masks:
           (1) the focal loss and the dice loss for fully supervised;
           (2) or the projection loss and pairwise affinity mask loss for weakly supervised loss.
         targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        gt_instances = kwargs['gt_instances']
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        # weakly supervised instance segmentation loss
        if self.boxinst_enabled:
            # filter samples only with empty instances
            new_gt_indices, new_gt_instances = [], []
            for indice, gt_instance in zip(indices, gt_instances):
                if not indice[1].numel():
                    continue
                new_gt_indices.append(indice[1])
                new_gt_instances.append(gt_instance)

            gt_inds = flat_gt_ids(new_gt_indices)

            # TODO: optimizing image with only backgroud
            if len(gt_inds) == 0:
                dummy_loss = src_masks.sum() * 0.
                return {"loss_prj"     : dummy_loss,
                        "loss_pairwise": dummy_loss}

            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in new_gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1)

            image_color_similarity = torch.cat([x.image_color_similarity for x in new_gt_instances])
            image_color_similarity = image_color_similarity[gt_inds]


            losses = self.boxinst_loss(src_masks, gt_bitmasks, image_color_similarity)
            return losses

        # fully supervised instance segmentation loss
        tgt_idx = self._get_tgt_permutation_idx(indices)
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(target_masks, point_coords, align_corners=False).squeeze(1)

        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_pseudo(self, outputs, outputs_ema, indices, num_masks, **kwargs):
        """Compute the losses related to the masks:
           (1) the focal loss and the dice loss for fully supervised;
           (2) or the projection loss and pairwise affinity mask loss for weakly supervised loss.
         targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        gt_instances = kwargs['gt_instances']
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        mask_scores = src_masks.sigmoid()

        # filter samples only with empty instances
        new_gt_indices, new_gt_instances = [], []
        for indice, gt_instance in zip(indices, gt_instances):
            if not indice[1].numel():
                continue
            new_gt_indices.append(indice[1])
            new_gt_instances.append(gt_instance)

        gt_inds = flat_gt_ids(new_gt_indices)

        # TODO: optimizing image with only backgroud
        if len(gt_inds) == 0:
            dummy_loss = src_masks.sum() * 0.
            return dummy_loss
        gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in new_gt_instances])
        gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1)

        masks = (mask_scores > 0.7) * gt_bitmasks.float()
        if self._iter > 0:
            pdb.set_trace()
            n_pos = torch.sum(masks, dim=(2,3))
            pseudo_scores = (mask_scores * gt_bitmasks).reshape(len(gt_inds), -1) 
            pseudo_scores_rank = torch.sort(pseudo_scores, descending=True)[0]
            n_pos = torch.clamp(n_pos, min=0, max=pseudo_scores_rank.shape[1]-1).long()
            
            assert n_pos.max() < pseudo_scores_rank.shape[1]
            thr = torch.gather(pseudo_scores_rank, dim=1, index=n_pos)
            thr = torch.clamp(thr, min=0.7, max=0.8)
            pseudo_seg_final = (mask_scores > thr) * gt_bitmasks.float()

            warmup_factor_2 = min(self._iter / float(self.max_iter), 1)
            weights = ((mask_scores > thr) | (mask_scores < 0.3)) * gt_bitmasks
            loss_pseudo = (self.mask_focal_loss(mask_scores, pseudo_seg_final.detach(), weights)) * warmup_factor_2 * 0.3
        return loss_pseudo
    
    def mask_focal_loss(x, targets, weights=None, alpha: float = 0.25, gamma: float = 2):
        ce_loss = F.binary_cross_entropy_with_logits(x, targets, weight=weights, reduction="none")
        p_t = x * targets + (1 - x) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.sum()/(weights.sum()+1)

    # def contrast_loss(self, weak_logits, strong_logits):
    #     """ the consistence loss ensures weak and strong data augmentation has the same outputs"""
    #     weak_logits = weak_logits.detach()
    #     min_thresh = self.consistency_cfg['mining']['min_thresh']
    #     max_thresh = self.consistency_cfg['mining']['max_thresh']
    #     pseudo_score = torch.sigmoid(weak_logits)

    #     if self.consistency_cfg['mining']['enabled']:
    #         bg = pseudo_score < min_thresh
    #         fg = pseudo_score > max_thresh
    #         m = bg | fg
    #     else:
    #         m = torch.ones_like(weak_logits)

    #     # compute pseudo label
    #     psuedo_label = pseudo_score.clone()
    #     if self.consistency_cfg['label_type'] == 'hard':
    #         psuedo_label[fg] = torch.tensor([1], dtype=psuedo_label.dtype, device=psuedo_label.device)
    #         psuedo_label[bg] = torch.tensor([0], dtype=psuedo_label.dtype, device=psuedo_label.device)

    #     # compute certainty loss
    #     if self.consistency_cfg['loss_type'] == 'ce':
    #         loss = F.binary_cross_entropy_with_logits(strong_logits, psuedo_label, reduction="none")
    #     elif self.consistency_cfg['loss_type'] == 'l1_smooth':
    #         loss = F.smooth_l1_loss(torch.sigmoid(strong_logits), torch.sigmoid(weak_logits), reduction="none")
    #     else:
    #         raise NotImplementedError
    #     loss = (m * loss).sum() / len(torch.nonzero(m)) if len(torch.nonzero(m)) else (m * loss).mean()

    #     # TODO: compute uncertainty loss
    #     return loss

  

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, gt_instances=None):
        loss_map = {
            'labels': self.loss_labels,
            'bboxes': self.loss_boxes,
            'masks' : self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks,
                              gt_instances=gt_instances)

    def split_weak_strong(self, outputs, targets, batch_inputs):
        def get_items_by_index(ids):
            items = dict()
            items['outputs'] = {
                'pred_logits': outputs['pred_logits'][ids],
                'pred_boxes' : outputs['pred_boxes'][ids],
                'pred_masks' : outputs['pred_masks'][ids],
                'aux_outputs': [{
                    'pred_logits': item['pred_logits'][ids],
                    'pred_boxes' : item['pred_boxes'][ids],
                    'pred_masks' : item['pred_masks'][ids]} for item in outputs['aux_outputs']]
            }
            items['batched_inputs'] = [batch_inputs[i] for i in ids]
            items['targets'] = [targets[i] for i in ids]
            return items

        weak_ids = list(range(0, len(targets), 2))
        strong_ids = list(range(1, len(targets), 2))
        weak_items = get_items_by_index(weak_ids)
        strong_items = get_items_by_index(strong_ids)
        return weak_items, strong_items

    def compute_losses(self, outputs, targets, batched_inputs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             batched_inputs: batched inputs
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        # TODO: Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        gt_instances = [x["instances"] for x in batched_inputs]
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, gt_instances=gt_instances))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, gt_instances=gt_instances)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # losses['loss_pseudo'] = 
        return losses

    def forward(self, outputs, outputs_ema, targets, batched_inputs, input_shape):
        assert len(targets) % 2 == 0
        assert outputs['pred_logits'].shape[0] % 2 == 0

        # pop ``masks`` for weakly supervised learning and get mask by bboxes
        if self.boxinst_enabled:
            for target in targets:
                target.pop('masks')
            self.boxinst_loss.compute_pseudo_masks(batched_inputs)

        # weak_items, strong_items = self.split_weak_strong(outputs, targets, batched_inputs)
        assert self.weak_aug or self.strong_aug, "weak_aug | strong_aug is False."

        losses = self.compute_losses(
                outputs,
                targets,
                batched_inputs,
            )
        return losses


    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
