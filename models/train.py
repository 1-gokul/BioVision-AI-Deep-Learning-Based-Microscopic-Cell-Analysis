"""
train.py - Fine-tune YOLOv8 on Blood Cell Detection dataset from Kaggle

Dataset: https://www.kaggle.com/datasets/drakeluo/blood-cell-detection-data-set
         OR: https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset

Both are in YOLO format already. Use the one that downloads successfully.

Usage:
    python models/train.py --data data/blood_cells/data.yaml --epochs 50

After training, the best weights are saved at:
    runs/detect/cell_detector/weights/best.pt

Copy that to: models/best.pt
"""

import argparse
import os
from ultralytics import YOLO


def train(data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 16, model_size: str = "n"):
    """
    Fine-tune YOLOv8 on cell data.

    Args:
        data_yaml: Path to dataset .yaml file
        epochs: Number of training epochs
        imgsz: Image size (640 is standard)
        batch: Batch size (reduce if GPU OOM)
        model_size: n/s/m/l/x (nano is fastest, x is most accurate)
    """
    model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="cell_detector",
        project="runs/detect",
        patience=15,           # Early stopping
        save=True,
        save_period=10,
        plots=True,
        val=True,
        device="0" if __import__("torch").cuda.is_available() else "cpu",
        workers=4,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        augment=True,
        mosaic=1.0,
        flipud=0.1,
        fliplr=0.5,
    )

    print("\n✅ Training complete!")
    print(f"Best weights: runs/detect/cell_detector/weights/best.pt")
    print("Copy it to models/best.pt and restart the app.\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on cell data")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"])
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ data.yaml not found at: {args.data}")
        print("Download the dataset first using: python models/download_dataset.py")
        exit(1)

    train(args.data, args.epochs, args.imgsz, args.batch, args.model)
