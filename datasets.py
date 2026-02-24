import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from proposals import generate_cone_proposals, box_iou_single
import cv2
from config import CROP_SIZE, IOU_EVAL_THRESH, IOU_NEG_THRESH, MAX_BG_PER_IMAGE

class ConeDataset(Dataset):
    def __init__(self, image_dir, label_dir, stems, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.stems = stems
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # find image file (jpg/png/jpeg)
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            p = os.path.join(self.image_dir, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"No image found for stem: {stem}")

        lbl_path = os.path.join(self.label_dir, stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        boxes, labels = [], []
        with open(lbl_path, "r") as f:
            for line in f:
                cls, xc, yc, w, h = map(float, line.split())
                x1 = (xc - w/2) * W
                y1 = (yc - h/2) * H
                x2 = (xc + w/2) * W
                y2 = (yc + h/2) * H

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # SHIFT for TorchVision detectors

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        img = self.transform(img)
        return img, target
    
class ProposalPatchDataset(Dataset):
    def __init__(self, base_dataset, crop_size=CROP_SIZE, iou_pos_thresh=IOU_EVAL_THRESH, iou_neg_thresh=IOU_NEG_THRESH, max_bg_per_image=MAX_BG_PER_IMAGE): #max_bg_pre_image was 20
        """
        base_dataset: your ConeDataset (returns img tensor, target)
        labels:
          0 = background
          1 = yellow cone
          2 = blue cone
        """
        self.base_dataset = base_dataset
        self.crop_size = crop_size
        self.iou_pos_thresh = iou_pos_thresh
        self.iou_neg_thresh = iou_neg_thresh
        self.max_bg_per_image = max_bg_per_image

        self.to_tensor = transforms.ToTensor()
        self.samples = []  # list of (image_idx, box, cls)

        self._build_index()

    def _build_index(self):
        for i in range(len(self.base_dataset)):
            img_tensor, target = self.base_dataset[i]
            # convert tensor [C,H,W] -> uint8 BGR for OpenCV
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            proposals = generate_cone_proposals(img_bgr)

            gt_boxes = target["boxes"]      # [N,4]
            gt_labels = target["labels"]    # [N] -> (1 yellow, 2 blue)

            bg_count = 0
            for p in proposals:
                ious = box_iou_single(p, gt_boxes)

                if len(ious) == 0:
                    # no GT in image -> background
                    if bg_count < self.max_bg_per_image:
                        self.samples.append((i, p, 0))
                        bg_count += 1
                    continue

                best_iou, best_idx = torch.max(ious, dim=0)
                best_iou = best_iou.item()
                best_idx = best_idx.item()

                if best_iou >= self.iou_pos_thresh:
                    cls = int(gt_labels[best_idx].item())  # 1 or 2
                    self.samples.append((i, p, cls))
                elif best_iou <= self.iou_neg_thresh and bg_count < self.max_bg_per_image:
                    self.samples.append((i, p, 0))
                    bg_count += 1
                # ignore ambiguous proposals between thresholds
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_idx, box, cls = self.samples[idx]
        img_tensor, _ = self.base_dataset[image_idx]

        # tensor [C,H,W] -> numpy RGB
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        x1, y1, x2, y2 = map(int, box)
        H, W, _ = img_np.shape

        # clip box to image bounds
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))

        crop = img_np[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)

        # Resize crop to classifier input size
        crop_pil = crop_pil.resize((self.crop_size, self.crop_size))

        crop_tensor = self.to_tensor(crop_pil)
        label = torch.tensor(cls, dtype=torch.long)

        return crop_tensor, label

def collate_fn(batch):
    return tuple(zip(*batch))  # images, targets