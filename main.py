import torch
import os

from model import ConeCNN
from inference import predict
from realtime import run_realtime
from config import NUM_CLASSES
from config import SCORE_THRESH, NMS_THRESH, IMAGE_DIR, WEIGHTS_PATH


def load_model(weights_path, device):
    model = ConeCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def run_image_test(model, device, visualize=False, image_paths = None):
    predict(
        model=model,
        image_paths=image_paths,
        device=device,
        visualize=visualize,
        score_thresh=SCORE_THRESH,
        nms_thresh=NMS_THRESH,
        save_dir = "Full_Pipeline_Results",
        save_dir_step1= "Segmentation_Results"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    weights_path = WEIGHTS_PATH
    if not os.path.exists(weights_path):
        print("Weights file not found:", weights_path)
        return

    model = load_model(weights_path, device)

    print("\nSelect mode:")
    print("1 - Test on image")
    print("2 - Real-time webcam")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("\nTesting on image. You can enter a single image path or multiple paths separated by commas or use the existed N samples in the dataset.")
        image_paths = input("Enter image path(s) or type the start of images you want to test from the samples in the dataset: ").strip()
        if image_paths.isdigit():
            end = input("Enter the end of images you want to test from the samples in the dataset: ").strip()
            if end.isdigit():
                start = int(image_paths)
                end = int(end)
                dataset_dir = IMAGE_DIR
                image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][start:end]
            else:
                print("Invalid end input. Please enter a valid number.")
                return
        run_image_test(model, device, image_paths=image_paths)

    elif choice == "2":
        run_realtime(model, device)

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()