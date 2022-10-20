import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy import random

from detectron2.structures import Boxes

from .box_ops import box_cxcywh_to_xyxy


def gen_color():
    return np.random.randint(0, 256, size=3)


def show(im, filename=None):
    fig, ax = plt.subplots()
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width, channels = im.shape
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if filename is None:
        plt.show()
    else:
        os.makedirs(filename.rsplit('/', 1)[0], exist_ok=True)
        plt.savefig(filename)


def plot_input_dict(dataset_dict):

    img = dataset_dict['image'].int().cpu().numpy().transpose(1, 2, 0).copy()
    if 'instances' in dataset_dict:
        bboxes = dataset_dict['instances'].gt_boxes.tensor.int().cpu().numpy()
        masks = dataset_dict['instances'].gt_masks.cpu().numpy()
        for bbox, mask in zip(bboxes, masks):
            x1, y1, x2, y2 = bbox
            color = gen_color()
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color.tolist(), thickness=2)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask = mask * color[None, None, :]
            img = cv2.addWeighted(img, 1.0, mask, 0.5, 0.)
        print(bboxes)
    show(img)


def plot_output_dict(batched_input, processed_result):
    img = batched_input['image'].int().cpu().numpy().transpose(1, 2, 0).copy()
    bboxes = processed_result['instances'].pred_boxes.tensor.int().cpu().numpy()

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    show(img)


def plot_network_input(batched_input, target):
    img = batched_input['image'].cpu().numpy().transpose(1, 2, 0).copy()
    w, h = img.shape[:2]

    boxes = Boxes(box_cxcywh_to_xyxy(target['boxes']))
    boxes.scale(scale_x=w, scale_y=h)
    boxes = boxes.tensor.cpu().numpy().astype(np.int)

    # boxes = box_cxcywh_to_xyxy(target['boxes']).cpu().numpy()
    # image_size_xyxy = np.array([[w, h, w, h]])
    # boxes = (boxes * image_size_xyxy).astype(np.int)

    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    show(img)

    print(boxes)


def visualize_color_similarity(img_color_sim_maps, num_max_ins=12):
    assert img_color_sim_maps.dim() == 3
    img_color_sim_maps = img_color_sim_maps.cpu().numpy()
    rows, cols = 2, 4
    num_maps = num_max_ins if num_max_ins < img_color_sim_maps.shape[0] else img_color_sim_maps.shape[0]
    assert num_maps == rows * cols

    fig, axes = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            if i * cols + j > num_maps:
                break
            axes[i, j].imshow(img_color_sim_maps[i * cols + j], cmap='gray')
            axes[i, j].axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.1, wspace=0.1)
    plt.margins(0, 0)
    plt.show()


def viz_boxes(image, boxes):
    img = image.int().cpu().numpy().transpose(1, 2, 0).copy()
    bboxes = boxes.tensor.int().cpu().numpy()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        color = gen_color()
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color.tolist(), thickness=2)
    show(img)


def viz_pesudo_masks(masks,  num_max_ins=12):
    masks = masks.int().cpu().numpy().copy()
    num_ins = masks.shape[0] if masks.shape[0] < num_max_ins else num_max_ins
    cols = 4
    rows = np.ceil(num_ins / cols).astype(int)

    fig, axes = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            if i * cols + j > num_ins - 1:
                break
            if rows > 1:
                axes[i, j].imshow(masks[i * cols + j], cmap='gray')
                axes[i, j].axis('off')
            else:
                axes[j].imshow(masks[i * cols + j], cmap='gray')
                axes[j].axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.1, wspace=0.1)
    plt.margins(0, 0)
    plt.show()


def viz_boxinst_masks(pred_masks, scores, img_prefix, score_thresh=0.5):
    valid_ids = torch.where(scores >= score_thresh)[0]

    for idx in valid_ids:
        im = pred_masks[idx].squeeze()
        fig, ax = plt.subplots()
        ax.imshow(im.cpu().numpy(), aspect='equal')
        plt.axis('off')
        height, width = im.shape
        fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        os.makedirs(img_prefix.rsplit('/', 1)[0], exist_ok=True)
        plt.savefig(f"{img_prefix}/mask-{idx.item()}.jpg")


def plot_color_similarity_on_image(input, color_sim_idx=0):
    from ..utils.box_ops import masks_to_boxes
    import torch.nn.functional as F

    pairwise_color_thresh = 0.3
    image = input['image'].cpu().numpy().transpose(1, 2, 0).copy()
    instances = input['instances']
    height, width = image.shape[:2]

    bitmasks = F.interpolate(instances.gt_bitmasks.unsqueeze(0), size=(height, width)).squeeze()
    image_color_similarity = F.interpolate(instances.image_color_similarity, size=(height, width))
    image_color_similarity = image_color_similarity[:, color_sim_idx, :, :].cpu().numpy()

    bboxes = masks_to_boxes(bitmasks.cpu()).numpy().astype(int)
    image = cv2.resize(image, (bitmasks.shape[-1], bitmasks.shape[-2]))
    bitmasks = bitmasks.cpu().numpy().astype(np.bool)
    show(image)
    for box, mask, color_sim in zip(bboxes, bitmasks, image_color_similarity):
        # color_sim = ((color_sim >= pairwise_color_thresh) * mask.cpu().numpy())
        color_sim = ((color_sim >= pairwise_color_thresh) | (1 - mask)).astype(np.uint8)
        color_sim = cv2.cvtColor(color_sim, cv2.COLOR_GRAY2RGB)
        image = image * color_sim
        color = gen_color()
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color.tolist(), thickness=2)
    show(image)