
WORKDIR=`dirname $0`

CONFIG=maskformer2_R50_bs16_50ep_weakly_baseline

python demo/demo.py \
  --config-file configs/coco/instance-segmentation/${CONFIG}.yaml \
  --input ${WORKDIR}/../../Data/test_images/**/*.jpg \
  --output ${WORKDIR}/../../Outputs/Mask2Former \
  --opts MODEL.WEIGHTS ${WORKDIR}/../../snapshots/Mask2Former/${CONFIG}/model_final.pth