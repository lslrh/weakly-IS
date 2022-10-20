WORKDIR=/apdcephfs/private_tobinwu/Research/Codes/Mask2Former
# OUTPUT_ROOT=/apdcephfs/private_tobinwu/Research/snapshots/Mask2Former
OUTPUT_ROOT=/youtu/xlab-team2-2/persons/tobinwu/Research/snapshots/Mask2Former
export DETECTRON2_DATASETS=/apdcephfs/private_tobinwu/Research/Data

CONFIG=maskformer2_R50_bs16_50ep_weakly_pair_v4_papxc_ce_soft

python3 train_net.py \
  --num-gpus=8 \
  --config-file=${WORKDIR}/configs/coco/instance-segmentation/${CONFIG}.yaml \
  --resume \
  OUTPUT_DIR ${OUTPUT_ROOT}/${CONFIG} \
  #  MODEL.WEIGHTS $WORKDIR/../../snapshots/Mask2Former/$CONFIG/model_0144999.pth