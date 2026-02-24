import torch
from torch.utils.data import DataLoader

from datasets import ConeDataset, ProposalPatchDataset
from model import ConeCNN
from evaluate import evaluate_full_pipeline, evaluate_proposal_recall
from config import *

import os
from sklearn.model_selection import train_test_split


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Prepare stems ----
    stems = [f.split(".")[0] for f in os.listdir(IMAGE_DIR)]
    train_stems, val_stems = train_test_split(stems, test_size=0.2, random_state=42)

    # ---- Base dataset ----
    train_dataset = ConeDataset(IMAGE_DIR, LABEL_DIR, train_stems)
    val_dataset   = ConeDataset(IMAGE_DIR, LABEL_DIR, val_stems)

    # ---- Patch dataset ----
    patch_train_dataset = ProposalPatchDataset(train_dataset)
    patch_val_dataset   = ProposalPatchDataset(val_dataset)

    patch_val_loader = DataLoader(
        patch_val_dataset,
        batch_size=BATCH_SIZE_PATCH,
        shuffle=False,
        num_workers=0
    )

    # ---- Model ----
    model = ConeCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    # ---- Evaluate full pipeline ----
    full_metrics = evaluate_full_pipeline(
        model=model,
        val_dataset=val_dataset,
        val_stems=val_stems,
        image_dir=IMAGE_DIR,
        device=device,
        iou_thresh=IOU_EVAL_THRESH,
        score_thresh=SCORE_THRESH,
        nms_thresh=NMS_THRESH,
        verbose=False
    )

    print("\n=== Full Pipeline (Segmentation + Classification + NMS) ===")
    print(
        f"Precision: {full_metrics['global']['precision']:.4f} | "
        f"Recall: {full_metrics['global']['recall']:.4f} | "
        f"F1: {full_metrics['global']['f1']:.4f} | "
        f"Mean IoU (TP): {full_metrics['global']['mean_iou_of_TP']:.4f}"
    )
    print(
        f"TP={full_metrics['global']['TP']} | "
        f"FP={full_metrics['global']['FP']} | "
        f"FN={full_metrics['global']['FN']}"
    )


    name_map = {1: "yellow", 2: "blue"}
    for cls_id in [1, 2]:
        m = full_metrics["per_class"][cls_id]
        print(
            f"{name_map[cls_id]:6s} -> "
            f"P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f} | "
            f"TP={m['TP']} FP={m['FP']} FN={m['FN']}"
        )

    # Evaluate proposal recall (without classification or NMS)
    evaluate_proposal_recall(val_dataset, val_stems, IMAGE_DIR, iou_thresh=0.3)


if __name__ == "__main__":
    main()