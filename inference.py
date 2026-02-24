import cv2
import torch
import torchvision.ops as ops
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from proposals import generate_cone_proposals
import os
from config import CROP_SIZE, SCORE_THRESH, NMS_THRESH, CLASS_NAME_MAP

def detect_cones(model, bgr_image, device,
                 crop_size=CROP_SIZE,
                 score_thresh=SCORE_THRESH,
                 nms_thresh=NMS_THRESH):

    model.eval()

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    proposals = generate_cone_proposals(bgr_image)

    if len(proposals) == 0:
        return [], [], [], []

    crops = []
    valid_boxes = []

    H, W, _ = rgb.shape
    to_tensor = transforms.ToTensor()

    for p in proposals:
        x1, y1, x2, y2 = map(int, p)

        # x1 = max(0, min(x1, W - 1))
        # y1 = max(0, min(y1, H - 1))
        # x2 = max(x1 + 1, min(x2, W))
        # y2 = max(y1 + 1, min(y2, H))

        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop).resize((crop_size, crop_size))
        crop_tensor = to_tensor(crop_pil)

        crops.append(crop_tensor)
        valid_boxes.append([x1, y1, x2, y2])

    if len(crops) == 0:
        return [], [], [], valid_boxes, [], []

    crops = torch.stack(crops).to(device)

    with torch.no_grad():
        logits = model(crops)
        probs = torch.softmax(logits, dim=1)
        scores, cls_ids = probs.max(dim=1)
        cls_ids_1 = cls_ids.clone().cpu().tolist()
        # probs_1 = probs.clone().cpu().tolist()
        scores_1 = scores.clone().cpu().tolist()
        # cls_prob_map = {}
        # for idx, cls_id in enumerate(cls_ids_1):
        #     cls_prob_map[idx] = probs_1[idx]
        #     print(f"Proposal {idx}: Predicted class {cls_id} with probabilities {probs_1[idx]}")

    keep = (cls_ids != 0) & (scores >= score_thresh)

    # if keep.sum().item() == 0:
    #     return [], [], [], valid_boxes, [], []

    boxes = torch.tensor(valid_boxes, dtype=torch.float32)[keep.cpu()]
    scores_kept = scores[keep].cpu()
    labels_kept = cls_ids[keep].cpu()

    keep_idx = ops.nms(boxes, scores_kept, nms_thresh)

    boxes = boxes[keep_idx].tolist()
    scores_kept = scores_kept[keep_idx].tolist()
    labels_kept = labels_kept[keep_idx].tolist()
    return boxes, labels_kept, scores_kept, valid_boxes, cls_ids_1, scores_1

def detect_cones_in_image(model, image_path, device,
                          crop_size=CROP_SIZE,
                          score_thresh=SCORE_THRESH,
                          nms_thresh=NMS_THRESH):

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    a,b,c,d,e,f = detect_cones(
    model,
    bgr,
    device,
    crop_size=crop_size,
    score_thresh=score_thresh,
    nms_thresh=nms_thresh
    )
    # print(f)
    return a,b,c,d,e,f

def detect_cones_in_frame(model, frame_bgr, device,
                          crop_size=CROP_SIZE,
                          score_thresh=SCORE_THRESH,
                          nms_thresh=NMS_THRESH):

    return detect_cones(
        model,
        frame_bgr,
        device,
        crop_size=crop_size,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh
    )

def draw_detections(image_path, boxes, labels, scores):
    img = cv2.imread(image_path)
    t = []
    for i, label in enumerate(labels):
        t.append((boxes[i], label, scores[i]))
    for box, cls, score in t:
        x1, y1, x2, y2 = map(int, box)
        
        name = CLASS_NAME_MAP.get(cls, "unknown")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{name}:{score:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    return img

def predict(
    model,
    image_paths,
    device,
    crop_size=CROP_SIZE,
    score_thresh=SCORE_THRESH,
    nms_thresh=NMS_THRESH,
    visualize=False,
    save_dir=None,            # full pipeline outputs (final boxes)
    save_dir_step1=None,      # ✅ new: step1 outputs (proposal boxes)
    return_images=False,
    return_step1_images=False # ✅ optional: return step1 images too
):
    # Allow single string input
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    results = []

    # create save dirs if needed
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if save_dir_step1 is not None:
        os.makedirs(save_dir_step1, exist_ok=True)

    for image_path in image_paths:
        # Run detection (now returns valid_boxes as 4th item)
        print(f"\nProcessing image: {image_path}")
        boxes, labels, scores, step1_boxes, step1_labels, step1_scores = detect_cones_in_image(
            model=model,
            image_path=image_path,
            device=device,
            crop_size=crop_size,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh
        )
        # Convert labels to names
        label_names = [CLASS_NAME_MAP.get(int(l), f"class_{l}") for l in labels]

        # Counts
        count_yellow = sum(1 for l in labels if int(l) == 1)
        count_blue = sum(1 for l in labels if int(l) == 2)
        count_total = len(labels)

        # Build result dict
        result = {
            "image_path": image_path,
            "boxes": boxes,
            "labels": labels,
            "label_names": label_names,
            "scores": scores,
            "count_total": count_total,
            "count_yellow": count_yellow,
            "count_blue": count_blue,
            "step1_boxes": step1_boxes
        }

        # ---- FULL PIPELINE VIS (final detections) ----
        vis_img = None
        if visualize or save_dir is not None or return_images:
            vis_img = draw_detections(image_path, boxes, labels, scores)

        # ---- STEP1 VIS (proposals only) ----
        step1_img = None
        if save_dir_step1 is not None or return_step1_images:
            step1_img = draw_detections(image_path, step1_boxes, step1_labels, step1_scores)
            # img = cv2.imread(image_path)
            if step1_img is not None:
            #     step1_img = img.copy()
            #     for b in (step1_boxes or []):
            #         x1, y1, x2, y2 = map(int, b)
            #         cv2.rectangle(step1_img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # yellow rectangles
                # optional label
                cv2.putText(step1_img, f"STEP1 proposals: {len(step1_boxes or [])}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Show image (full pipeline)
        if visualize and vis_img is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title(f"{os.path.basename(image_path)} | total={count_total}, yellow={count_yellow}, blue={count_blue}")
            plt.axis("off")
            plt.show()

        # Save full pipeline image
        if save_dir is not None and vis_img is not None:
            out_name = os.path.basename(image_path)
            out_path = os.path.join(save_dir, out_name)
            cv2.imwrite(out_path, vis_img)
            result["saved_path"] = out_path

        # Save step1 image
        if save_dir_step1 is not None and step1_img is not None:
            out_name = os.path.basename(image_path)
            out_path = os.path.join(save_dir_step1, out_name)
            cv2.imwrite(out_path, step1_img)
            result["saved_step1_path"] = out_path

        # Optionally return image arrays
        if return_images and vis_img is not None:
            result["vis_image"] = vis_img
        if return_step1_images and step1_img is not None:
            result["step1_image"] = step1_img

        results.append(result)

    return results