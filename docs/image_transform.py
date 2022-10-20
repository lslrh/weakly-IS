import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.data import transforms as T


img_path = "test_images/Bewick_Wren_0113_185113.jpg"

augs = T.AugmentationList(
    [
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(),
        T.FixedSizeCrop((800, 800), pad=True),
        # T.RandomCrop(),
    ]
)


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("image shape:", img.shape)
    return img


def load_annotations(anno_path):
    """load anntations

    Args:
        anno_path (str): path of annotation file

    Returns:
        boxes (ndarray): N*4
        masks (ndarray): N*H*W
    """
    with open(anno_path, "r") as f:
        infos = json.load(f)

    boxes, polygons = [], []
    for item in infos["shapes"]:
        if item["shape_type"] == "rectangle":
            x1, y1 = item["points"][0]
            x2, y2 = item["points"][1]
            bbox = [x1, y1, x2, y2]
            boxes.append(bbox)
            continue
        if item["shape_type"] == "polygon":
            polygon = item["points"]
            polygons.append(polygon)
            continue
    boxes = np.array(boxes, dtype=np.int32)
    polygons = np.array(polygons, dtype=np.int32)
    print("boxes   shape: ", boxes.shape)
    print("polygon shape:", polygons.shape)

    # polygon to mask
    masks = np.zeros((len(polygons), *img.shape[:2]), dtype=np.uint8)
    for i, polygon in enumerate(polygons):
        print(masks[i].shape)
        # cv2.polylines(masks[i], polygon, 120, 1)
        cv2.fillPoly(masks[i], [polygon], 255, 1)
    return {"boxes": boxes, "masks": masks}


img = load_image(img_path)
anns = load_annotations(img_path.replace("jpg", "json"))


plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(anns["masks"][0], cmap="gray")
plt.show()


input = T.AugInput(img, boxes=anns["boxes"], sem_seg=anns["masks"][0])
transform = augs(input)
img_t = input.image
boxes_t = input.boxes
masks_t = input.sem_seg

boxes_t = np.maximum(0, boxes_t)
boxes_t[:, 2] = np.minimum(boxes_t[:, 2], img_t.shape[1])
boxes_t[:, 3] = np.minimum(boxes_t[:, 3], img_t.shape[0])
print("transformed image shape:", img_t.shape)
print("transformed bboxes:", boxes_t)

img_t_view = img_t.copy()
plt.subplot(1, 2, 1)
for box in boxes_t:
    cv2.rectangle(img_t_view, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
plt.imshow(img_t_view)
plt.subplot(1, 2, 2)
plt.imshow(masks_t, cmap="gray")
plt.show()


inv_transform = transform.inverse()
print(inv_transform)
inv_img = inv_transform.apply_image(img_t)
inv_masks = inv_transform.apply_segmentation(masks_t)
inv_boxes = inv_transform.apply_box(boxes_t)

plt.subplot(1, 2, 1)
for box in inv_boxes:
    cv2.rectangle(inv_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
plt.imshow(inv_img)
plt.subplot(1, 2, 2)
plt.imshow(inv_masks, cmap="gray")
plt.show()