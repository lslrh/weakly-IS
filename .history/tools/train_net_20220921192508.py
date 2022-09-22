# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"
from collections import OrderedDict
import torch
import copy
import itertools
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from detectron2.solver.build import maybe_add_gradient_clipping

from adet.data import build_voc_detection_train_loader, build_detection_train_loader, load_coco_json
from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.data.dataset_strong_weak_aug_mapper import COCOInstancePairOverlapDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator
from typing import Any, Dict, List, Set

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.coco import load_coco_json

TRAIN_PATH = '/home/liruihuang/weakly-IS/datasets/VOC2012/train'
TRAIN_JSON = '/home/liruihuang/weakly-IS/datasets/VOC2012/annotations/voc_2012_train.json'
VAL_PATH = '/home/liruihuang/weakly-IS/datasets/VOC2012/val'
VAL_JSON = '/home/liruihuang/weakly-IS/datasets/VOC2012/annotations/voc_2012_val.json'
CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

import pdb

class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    # @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.MODEL.FCPOSE_ON:
            mapper = FCPoseDatasetMapper(cfg, True)
        else:
            # mapper = COCOInstancePairOverlapDatasetMapper(cfg, True)
            mapper = DatasetMapperWithBasis(cfg, True)
        return build_voc_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    # @classmethod
    # def build_optimizer(cls, cfg, model):
    #     weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    #     weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    #     defaults = {}
    #     defaults["lr"] = cfg.SOLVER.BASE_LR
    #     defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    #     norm_module_types = (
    #         torch.nn.BatchNorm1d,
    #         torch.nn.BatchNorm2d,
    #         torch.nn.BatchNorm3d,
    #         torch.nn.SyncBatchNorm,
    #         # NaiveSyncBatchNorm inherits from BatchNorm2d
    #         torch.nn.GroupNorm,
    #         torch.nn.InstanceNorm1d,
    #         torch.nn.InstanceNorm2d,
    #         torch.nn.InstanceNorm3d,
    #         torch.nn.LayerNorm,
    #         torch.nn.LocalResponseNorm,
    #     )

    #     params: List[Dict[str, Any]] = []
    #     memo: Set[torch.nn.parameter.Parameter] = set()
    #     for module_name, module in model.named_modules():
    #         for module_param_name, value in module.named_parameters(recurse=False):
    #             if not value.requires_grad:
    #                 continue
    #             # Avoid duplicating parameters
    #             if value in memo:
    #                 continue
    #             memo.add(value)

    #             hyperparams = copy.copy(defaults)
    #             if "backbone" in module_name:
    #                 hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
    #             if (
    #                 "relative_position_bias_table" in module_param_name
    #                 or "absolute_pos_embed" in module_param_name
    #             ):
    #                 print(module_param_name)
    #                 hyperparams["weight_decay"] = 0.0
    #             if isinstance(module, norm_module_types):
    #                 hyperparams["weight_decay"] = weight_decay_norm
    #             if isinstance(module, torch.nn.Embedding):
    #                 hyperparams["weight_decay"] = weight_decay_embed
    #             params.append({"params": [value], **hyperparams})

        # def maybe_add_full_model_gradient_clipping(optim):
        #     # detectron2 doesn't have full model gradient clipping now
        #     clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        #     enable = (
        #         cfg.SOLVER.CLIP_GRADIENTS.ENABLED
        #         and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
        #         and clip_norm_val > 0.0
        #     )

        #     class FullModelGradientClippingOptimizer(optim):
        #         def step(self, closure=None):
        #             all_params = itertools.chain(*[x["params"] for x in self.param_groups])
        #             torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
        #             super().step(closure=closure)

        #     return FullModelGradientClippingOptimizer if enable else optim

        # optimizer_type = cfg.SOLVER.OPTIMIZER
        # if optimizer_type == "SGD":
        #     optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
        #         params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        #     )
        # elif optimizer_type == "ADAMW":
        #     optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
        #         params, cfg.SOLVER.BASE_LR
        #     )
        # else:
        #     raise NotImplementedError(f"no optimizer type {optimizer_type}")
        # if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        #     optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        # return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def plain_register_dataset():
    DatasetCatalog.register("voc_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("voc_train").set(thing_classes=CLASS_NAMES, 
                                                    evaluator_type='coco', 
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)

    #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
    DatasetCatalog.register("voc_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("voc_val").set(thing_classes=CLASS_NAMES, 
                                                evaluator_type='coco', 
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)

def main(args):
    cfg = setup(args)
    plain_register_dataset()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
