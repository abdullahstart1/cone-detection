# config.py

import os
import numpy as np

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
IMAGE_DIR = os.path.join(BASE_DIR, "cones_dataset", "images")
LABEL_DIR = os.path.join(BASE_DIR, "cones_dataset", "labels")
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "cone_cnn_patch_classifier.pth")

# Patch classifier parameters
CROP_SIZE = 64
NUM_CLASSES = 3  # 0=background, 1=yellow, 2=blue

# Detection thresholds
SCORE_THRESH = 0.2
NMS_THRESH = 0.9
IOU_EVAL_THRESH = 0.5
IOU_NEG_THRESH = 0.2
MAX_BG_PER_IMAGE = 80

# Training hyperparameters
BATCH_SIZE_PATCH = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

# BLUE_LOWER = np.array([85, 50, 20])
# BLUE_UPPER = np.array([140, 255, 255])
BLUE_LOWER = np.array([85, 5, 20])
BLUE_UPPER = np.array([140, 255, 255])
YELLOW_LOWER = np.array([15, 50, 20])
YELLOW_UPPER = np.array([45, 255, 255])

MIN_CONTOUR_AREA = 20
MIN_HEIGHT = 6
MIN_WIDTH = 3
PAD_X_FACTOR = 0.05
PAD_Y_FACTOR = 0.05

# Class name mapping
CLASS_NAME_MAP = {
    0: "bg",
    1: "y",
    2: "b"
}