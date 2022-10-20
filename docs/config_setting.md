

### Pycharm
#### Training

```text
--config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_with_bbox.yaml
 OUTPUT_DIR /apdcephfs/private_tobinwu/Research/snapshots/Mask2Former/debug 
 MODEL.WEIGHTS /apdcephfs/private_tobinwu/pretrained/detectron2/ImageNetPretrained/torchvision/R-50.pkl 
 SOLVER.IMS_PER_BATCH 1
 INPUT.IMAGE_SIZE 512 
 DATALOADER.NUM_WORKERS 0
```

#### Evaluation

```text
--config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_with_bbox.yaml
--eval-only
MODEL.WEIGHTS /apdcephfs/private_tobinwu/Research/snapshots/Mask2Former/maskformer2_R50_bs16_50ep_with_bbox_1/model_0019999.pth
OUTPUT_DIR /apdcephfs/private_tobinwu/Research/snapshots/Mask2Former/maskformer2_R50_bs16_50ep_with_bbox
```

### Demo

> 

```text
--config-file ../configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_with_bbox_fixed_size.yaml
--input /apdcephfs/private_tobinwu/Research/Data/test_images/birds/*.jpg
--output /apdcephfs/private_tobinwu/Research/Outputs/Mask2Former
--opts
MODEL.WEIGHTS
/apdcephfs/private_tobinwu/Research/snapshots/Mask2Former/maskformer2_R50_bs16_50ep_with_bbox_1/model_0014999.pth
```

> BoxInst Loss
```text
--config-file ../configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_weakly_baseline.yaml
--input /apdcephfs/private_tobinwu/Research/Data/test_images/birds/*.jpg
--output /apdcephfs/private_tobinwu/Research/Outputs/Mask2Former
--opts
MODEL.WEIGHTS
/apdcephfs/private_tobinwu/Research/snapshots/Mask2Former/maskformer2_R50_bs16_50ep_weakly_baseline/model_0169999.pth
```