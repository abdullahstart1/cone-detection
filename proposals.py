import cv2
import numpy as np
import torch
from config import BLUE_LOWER, BLUE_UPPER, YELLOW_LOWER, YELLOW_UPPER, MIN_CONTOUR_AREA, MIN_HEIGHT, MIN_WIDTH, PAD_X_FACTOR, PAD_Y_FACTOR

def generate_cone_proposals(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Example ranges (you must tune them on your dataset)
    blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    open_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, open_kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, close_kernel)

    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, open_kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel)

    proposals = []

    for mask in [blue_mask, yellow_mask]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL to get only outer contours
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:   # tune this
                continue
            x, y, w, h = cv2.boundingRect(c)

            # Optional shape filtering
            if h < MIN_HEIGHT or w < MIN_WIDTH:
                continue

            H, W = image_bgr.shape[:2]

            x1 = max(0, x - w * PAD_X_FACTOR)
            y1 = max(0, y - h * PAD_Y_FACTOR)
            x2 = min(W, x + w + w * PAD_X_FACTOR)
            y2 = min(H, y + h + h * PAD_Y_FACTOR)

            proposals.append([x1, y1, x2, y2])
    return proposals

def box_iou_single(box, boxes):
    """
    box: [x1, y1, x2, y2]
    boxes: tensor [N, 4]
    returns IoU tensor [N]
    """
    if len(boxes) == 0:
        return torch.tensor([])

    box = torch.tensor(box, dtype=torch.float32)

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)
    inter = inter_w * inter_h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter + 1e-6
    return inter / union

def compute_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union