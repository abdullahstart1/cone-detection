import torch
import numpy as np
import os
import cv2
from inference import detect_cones_in_image
from proposals import compute_iou, generate_cone_proposals
from config import NUM_CLASSES, IOU_EVAL_THRESH, SCORE_THRESH, NMS_THRESH, CROP_SIZE

def find_image_path(image_dir, stem):
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        p = os.path.join(image_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def evaluate_patch_classifier(model, loader, device, criterion):
    model.eval()

    total = 0
    correct = 0
    running_loss = 0.0

    # classes: 0=background, 1=yellow, 2=blue
    num_classes = NUM_CLASSES
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # rows=true, cols=pred

    with torch.no_grad():
        for imgs, labels in loader:
            # handle both tensor batch and tuple/list batch
            imgs = torch.stack(imgs).to(device) if isinstance(imgs, (tuple, list)) else imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_mat[t.long(), p.long()] += 1

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)

    # Per-class precision/recall/F1
    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = conf_mat[c, c].item()
        fp = conf_mat[:, c].sum().item() - tp
        fn = conf_mat[c, :].sum().item() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Macro metrics (all classes)
    macro_precision = sum(precisions) / num_classes
    macro_recall    = sum(recalls) / num_classes
    macro_f1        = sum(f1s) / num_classes

    # Cone-only macro metrics (exclude background class 0)
    cone_macro_precision = sum(precisions[1:]) / 2
    cone_macro_recall    = sum(recalls[1:]) / 2
    cone_macro_f1        = sum(f1s[1:]) / 2

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "per_class": per_class,  # 0:bg, 1:yellow, 2:blue
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "cone_macro_precision": cone_macro_precision,
        "cone_macro_recall": cone_macro_recall,
        "cone_macro_f1": cone_macro_f1,
        "confusion_matrix": conf_mat
    }


def evaluate_full_pipeline(
    model,
    val_dataset,   # ConeDataset (labels already shifted: 1=yellow, 2=blue)
    val_stems,
    image_dir,
    device,
    iou_thresh=IOU_EVAL_THRESH,
    score_thresh= SCORE_THRESH,
    nms_thresh= NMS_THRESH,
    verbose=False
):
    model.eval()

    # global counts
    TP = 0
    FP = 0
    FN = 0

    # per-class counts: 1=yellow, 2=blue
    per_class = {
        1: {"TP": 0, "FP": 0, "FN": 0},
        2: {"TP": 0, "FP": 0, "FN": 0},
    }

    matched_ious = []  # IoUs of true positives only

    for i in range(len(val_dataset)):
        _, target = val_dataset[i]
        gt_boxes = target["boxes"].cpu().numpy().tolist()
        gt_labels = target["labels"].cpu().numpy().tolist()

        stem = val_stems[i]
        img_path = find_image_path(image_dir, stem)
        if img_path is None:
            continue

        pred_boxes, pred_labels, pred_scores, _, _, _ = detect_cones_in_image(
            model=model,
            image_path=img_path,
            device=device,
            crop_size=CROP_SIZE,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh
        )

        # Sort predictions by score descending (greedy matching)
        order = sorted(range(len(pred_boxes)), key=lambda k: pred_scores[k], reverse=True)
        pred_boxes  = [pred_boxes[k] for k in order]
        pred_labels = [pred_labels[k] for k in order]
        pred_scores = [pred_scores[k] for k in order]

        matched_gt = [False] * len(gt_boxes)

        # Match each prediction to best GT of same class
        for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if matched_gt[g_idx]:
                    continue
                if pl != gl:
                    continue

                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if verbose:
                print(f"[{stem}] pred_label={pl} score={ps:.3f} best_iou={best_iou:.3f}")

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                TP += 1
                per_class[pl]["TP"] += 1
                matched_gt[best_gt_idx] = True
                matched_ious.append(best_iou)
            else:
                FP += 1
                if pl in per_class:
                    per_class[pl]["FP"] += 1

        # Any unmatched GT is a false negative
        for g_idx, was_matched in enumerate(matched_gt):
            if not was_matched:
                FN += 1
                gl = gt_labels[g_idx]
                if gl in per_class:
                    per_class[gl]["FN"] += 1

    # Global metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou_tp = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    # Per-class metrics
    per_class_metrics = {}
    for cls_id, counts in per_class.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_class_metrics[cls_id] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "TP": tp,
            "FP": fp,
            "FN": fn
        }

    return {
        "global": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "mean_iou_of_TP": mean_iou_tp,
            "num_matched_tp": len(matched_ious)
        },
        "per_class": per_class_metrics,  # 1=yellow, 2=blue
        "matched_ious": matched_ious
    }

def evaluate_proposal_recall(val_dataset, val_stems, image_dir, iou_thresh=0.3):
    total_gt = 0
    covered_gt = 0

    for i in range(len(val_dataset)):
        img_path = find_image_path(image_dir, val_stems[i])
        if img_path is None:
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            continue

        proposals = generate_cone_proposals(bgr)

        _, target = val_dataset[i]
        gt_boxes = target["boxes"].cpu().numpy().tolist()

        total_gt += len(gt_boxes)

        for gb in gt_boxes:
            found = False
            for pb in proposals:
                if compute_iou(pb, gb) >= iou_thresh:
                    found = True
                    break
            if found:
                covered_gt += 1

    recall = covered_gt / total_gt if total_gt > 0 else 0.0
    print(f"Proposal Recall @IoU>={iou_thresh}: {recall:.4f} ({covered_gt}/{total_gt})")
    return recall