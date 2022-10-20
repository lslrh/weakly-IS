# Copyright (c) Tencent, Inc. and its affiliates.
# Modified by Tao Wu from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import random

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, Boxes, PolygonMasks

from pycocotools import mask as coco_mask

__all__ = ["COCOInstancePairDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def build_crop_gen(cfg, is_train):
    if cfg.INPUT.CROP.ENABLED and is_train:
        crop_gen = [
            T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
            T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
        ]
    else:
        crop_gen = None
    return crop_gen


def build_strong_transform_gen(cfg, is_train):
    if is_train:
        tfms = [
            T.RandomBrightness(intensity_min=0.5, intensity_max=2.),
            T.RandomLighting(scale=0.2),
            T.RandomSaturation(intensity_min=0.5, intensity_max=2.),
            T.RandomContrast(intensity_min=0.3, intensity_max=20),
        ]
    return tfms


# This is specifically designed for the COCO dataset.
class COCOInstancePairDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
            self,
            is_train=True,
            *,
            tfm_gens,
            crop_gens,
            strong_gens,
            image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        self.crop_gens = crop_gens
        self.strong_gens = strong_gens

        self.img_format = image_format
        self.is_train = is_train

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        crop_gens = build_crop_gen(cfg, is_train)
        strong_gens = build_strong_transform_gen(cfg, is_train)

        ret = {
            "is_train"    : is_train,
            "tfm_gens"    : tfm_gens,
            "crop_gens"   : crop_gens,
            "strong_gens" : strong_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def process_data(self, dataset_dict, image, transforms):

        result_dict = copy.deepcopy(dataset_dict)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # therefore it's important to use torch.Tensor.
        result_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Implement additional transformations if you have other types of data
        annos = []
        for obj in dataset_dict['annotations']:
            if obj.get("iscrowd", 0) == 0:
                annos.append(utils.transform_instance_annotations(obj, transforms, image_shape))

        # NOTE: does not support BitMask due to augmentation
        # Current BitMask cannot handle empty objects
        instances = utils.annotations_to_instances(annos, image_shape)

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if hasattr(instances, 'gt_masks'):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        # Need to filter empty instances first (due to augmentation)
        instances = utils.filter_empty_instances(instances)

        # Generate masks from polygon
        h, w = instances.image_size
        if hasattr(instances, 'gt_masks'):
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks
        else:
            instances.gt_masks = torch.zeros((0, *image_shape))

        result_dict["instances"] = instances

        return result_dict

    def weak_transform(self, dataset_dict, image):
        """weak augmentation"""
        if self.crop_gens is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gens + self.tfm_gens[-1:], image
                )
        return self.process_data(dataset_dict, image, transforms), image

    def strong_transform(self, dataset_dict, image):
        """strong transforms, without geometric transform"""
        strong_gens = random.sample(self.strong_gens, 2)
        image, transforms = T.apply_transform_gens(strong_gens, image)
        return self.process_data(dataset_dict, image, transforms), image

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        result_weak, image = self.weak_transform(dataset_dict, image)
        result_strong, _ = self.strong_transform(dataset_dict, image)

        return result_weak, result_strong
