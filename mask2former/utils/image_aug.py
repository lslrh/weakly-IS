import random
import copy
import torch
from detectron2.structures import Boxes, Instances
import torch.nn as nn
import kornia.augmentation as K
from kornia.augmentation.container import AugmentationSequential


class StrongAugmentationPipeline(nn.Module):
    def __init__(self) -> None:
        super(StrongAugmentationPipeline, self).__init__()
        self.aug_list = AugmentationSequential(
            K.ColorJitter(0.2, 0.2, 0.2, 0.2, p=1.),
            # K.RandomEqualize(p=0.5),
            # K.RandomGrayscale(),
            # K.RandomBoxBlur(keepdim=True),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5), p=1.),
            K.RandomGaussianNoise(),
            # K.RandomMotionBlur(kernel_size=(3, 18), angle=90, direction=0., keepdim=True),
            data_keys=['input'],
            random_apply=2,
        )

    @torch.no_grad()
    def forward(self, input):
        image = copy.deepcopy(input['image'].float())
        # bboxes = input['instances'].gt_boxes.tensor
        try:
            res = self.aug_list(image)
        except:
            print('s')

        # format output
        image_size = res[0].shape[-2:]
        instances = Instances(image_size)
        instances.gt_boxes = input['instances'].gt_boxes
        instances.gt_classes = input['instances'].gt_classes
        instances.gt_masks = input['instances'].gt_masks
        output = dict(
            filename=input['file_name'],
            height=input['height'],
            width=input['width'],
            image_id=input['image_id'],
            image=res[0].squeeze(),
            instances=instances
        )

        # output = copy.deepcopy(input)
        # output['image'] = res[0].squeeze()
        # output['instances'].gt_boxes = Boxes(res[1])

        assert image.shape == output['image'].shape

        return output
