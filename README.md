# Cone Detection System

This repository implements a **cone detection system** that uses **image processing** and **machine learning** to identify **yellow** and **blue cones** in images. The system uses **HSV color space**, **contour detection**, and a **Convolutional Neural Network (CNN)** to classify cones from images. It also applies **Non-Maximum Suppression (NMS)** to remove redundant detections.

## Overview

The cone detection system is built to identify **yellow** and **blue cones** using computer vision techniques, specifically focusing on image segmentation and classification. The system works by first detecting regions of interest (proposals) using **HSV color filtering**, followed by **contour detection**. These regions are then classified using a **CNN**. Finally, **Non-Maximum Suppression (NMS)** is applied to remove overlapping bounding boxes.

### Key Features:
- **HSV-based color detection** for yellow and blue cones.
- **Contour detection** to find cone candidates.
- **Convolutional Neural Network (CNN)** for classification.
- **Non-Maximum Suppression (NMS)** to filter redundant detections.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdullahstart1/cone-detection.git
   cd cone-detection

2. **Create a virtual environment (optional but recommended):**

    python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies:**

    pip install -r requirements.txt

4. **Download or prepare your dataset. The dataset should include:**

    images folder: Contains the input images with cones.

    labels folder: Contains the corresponding ground truth annotations for cone locations.

    If you don't have a dataset, you can use your own or collect one.


## Model Training

To train the CNN model for cone classification, follow these steps:

Prepare the dataset in the following format:

images folder: Contains images of cones.

labels folder: Contains the bounding box annotations for each image

Run the training script:

python train.py

Modify the configuration file (config.py) to set the correct paths for the dataset, as well as other hyperparameters like learning rate, batch size, etc.

The model will be trained and saved to the weights directory.

## Testing the Model

Once you have a trained model, you can test it on new images using the following steps:

Run the image testing script:

python main.py

Choose mode:

1: Test on a single image.

2: Real-time webcam feed.

Enter the image path or select from the available dataset. The system will output the bounding boxes and class labels for each detected cone in the image.

Example Command for Testing:
python main.py
Select mode:

1 - Test on image

2 - Real-time webcam

Enter choice (1 or 2): 1
Enter image path(s) or type the start of images you want to test from the samples in the dataset: 0

Enter the end of images you want to test from the samples in the dataset: 40


## Evaluation

You can evaluate the model's performance on the validation dataset using the following command:

python evaluate_main.py

This will compute evaluation metrics such as:

- Precision
- Recall
- F1 Score
- Mean Intersection over Union (IoU)

The results will be displayed in the terminal.


### Project Structure
```
/cone-detection
    ├── cones_dataset/            # Dataset directory
    │   ├── images/               # Folder for image files
    │   ├── labels/               # Folder for label files
    │   └── config.yaml           # Dataset configuration file
    ├── weights/                  # Model weights directory
    │   └── cone_cnn_patch_classifier.pth        #Model weights
    ├── config.py                 # Configuration file
    ├── datasets.py               # Dataset-related Python file
    ├── evaluate.py               # Evaluation script
    ├── evaluate_main.py          # Main evaluation script
    ├── inference.py              # Inference script
    ├── main.py                   # Main application file
    ├── model.py                  # Model architecture script
    ├── proposals.py              # Proposal generation code
    ├── realtime.py               # Real-time processing code
    ├── requirements.txt          # Python dependencies
    ├── train.py                  # Training script
    ├── train_main.py             # Main training script
    └──── README.md               # Project description in markdown
