# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_pair_overlap_dataset_mapper import COCOInstancePairOverlapDatasetMapper
from .data.dataset_mappers.coco_instance_pair_dataset_mapper import COCOInstancePairDatasetMapper
from .data.dataset_mappers.coco_instance_detr_type_dataset_mapper import COCOInstanceDetrTypeDatasetMapper
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import MaskFormerInstanceDatasetMapper
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import MaskFormerPanopticDatasetMapper
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
