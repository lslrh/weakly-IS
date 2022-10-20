WORKDIR=/apdcephfs/private_tobinwu/Research/Codes/Mask2Former

export DETECTRON2_DATASETS=/apdcephfs/private_tobinwu/Research/Data

CONFIG=maskformer2_R50_bs16_50ep_weakly_baseline_affinity_weight=0.1

python train_net.py \
  --num-gpus=1 \
  --config-file=${WORKDIR}/configs/coco/instance-segmentation/${CONFIG}.yaml \
  --eval-only \
  MODEL.WEIGHTS ${WORKDIR}/../../snapshots/Mask2Former/${CONFIG}/model_0339999.pth \
  OUTPUT_DIR /apdcephfs/private_tobinwu/Research/snapshots/Mask2Former/${CONFIG}
