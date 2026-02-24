import torch
from torch.utils.data import DataLoader

from datasets import ConeDataset, ProposalPatchDataset
from model import ConeCNN
from train import train
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

    patch_train_loader = DataLoader(
        patch_train_dataset,
        batch_size=BATCH_SIZE_PATCH,
        shuffle=True,
        num_workers=0
    )

    patch_val_loader = DataLoader(
        patch_val_dataset,
        batch_size=BATCH_SIZE_PATCH,
        shuffle=False,
        num_workers=0
    )

    # ---- Model ----
    model = ConeCNN(num_classes=NUM_CLASSES).to(device)

    # ---- Train ----
    train(
        model,
        patch_train_loader,
        patch_val_loader,
        device,
        save_path="weights/cone_cnn_patch_classifier.pth"
    )


if __name__ == "__main__":
    main()