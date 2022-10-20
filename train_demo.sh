WORKDIR=/apdcephfs/private_tobinwu/Research/Codes/Mask2Former

export DETECTRON2_DATASETS=/apdcephfs/private_tobinwu/Research/Data

CONFIG=maskformer2_R50_bs16_50ep_weakly_pair_paxxc

python3 -m torch.utils.bottleneck train_net.py \
  --config-file \
  configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_weakly_pair_paxxc.yaml \
  OUTPUT_DIR \
  /apdcephfs/private_tobinwu/Research/snapshots/Mask2Former/debug \
  SOLVER.IMS_PER_BATCH 1 \
  INPUT.IMAGE_SIZE 512 \
  INPUT.MIN_SIZE_TRAIN 360, \
  DATALOADER.NUM_WORKERS 0