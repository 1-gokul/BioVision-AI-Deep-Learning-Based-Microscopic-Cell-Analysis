"""
cell_detector.py - Core detection logic
Uses YOLOv8 for cell detection and classification.
Falls back to OpenCV contour detection if no trained model is available.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Class labels matching the Blood Cell Detection dataset
CLASS_NAMES = {
    0: "RBC",       # Red Blood Cell - healthy
    1: "WBC",       # White Blood Cell - healthy
    2: "Platelets", # Platelets - healthy
}

# Which classes are considered "abnormal" — extend this as you add more classes
ABNORMAL_CLASSES = set()  # In base dataset all are normal; extend for pathology datasets

HEALTHY_COLOR = (0, 255, 0)    # Green
ABNORMAL_COLOR = (0, 0, 255)   # Red


@dataclass
class CellResult:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    is_abnormal: bool
    area: float


def load_model(model_path: Optional[str] = None):
    """
    Load YOLOv8 model.
    If a custom trained model path is given, load it.
    Otherwise load YOLOv8n pretrained (you'll fine-tune this).
    """
    try:
        from ultralytics import YOLO
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"Loaded custom model from {model_path}")
        else:
            # Use nano model as base; replace with fine-tuned weights after training
            model = YOLO("yolov8n.pt")
            print("Loaded YOLOv8n base model (not fine-tuned on cell data yet)")
        return model
    except Exception as e:
        print(f"Could not load YOLO: {e}")
        return None


def detect_with_yolo(image: np.ndarray, model, conf_threshold: float = 0.3) -> List[CellResult]:
    """Run YOLOv8 inference and parse results."""
    results = model(image, conf=conf_threshold, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            is_abnormal = cls_id in ABNORMAL_CLASSES
            area = (x2 - x1) * (y2 - y1)

            detections.append(CellResult(
                class_name=class_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                is_abnormal=is_abnormal,
                area=area
            ))

    return detections


def detect_with_opencv_fallback(image: np.ndarray) -> List[CellResult]:
    """
    OpenCV contour-based cell detection.
    Used as a fallback when no trained YOLO model is available.
    Also useful for comparing results.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Adaptive threshold works better than fixed for microscopy images
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    h, w = image.shape[:2]
    min_area = (h * w) * 0.0005  # At least 0.05% of image
    max_area = (h * w) * 0.15    # At most 15% of image

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        # Aspect ratio filter: cells are roughly circular
        aspect = bw / bh if bh > 0 else 0
        if aspect < 0.3 or aspect > 3.0:
            continue

        # Circularity check
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Classify by size heuristic (rough proxy without trained model)
        cell_size_ratio = area / (h * w)
        if cell_size_ratio > 0.02:
            class_name = "WBC"
        elif circularity > 0.7:
            class_name = "RBC"
        else:
            class_name = "Platelet"

        detections.append(CellResult(
            class_name=class_name,
            confidence=round(circularity, 2),  # Use circularity as proxy confidence
            bbox=(float(x), float(y), float(x + bw), float(y + bh)),
            is_abnormal=False,
            area=float(area)
        ))

    return detections


def draw_detections(image: np.ndarray, detections: List[CellResult]) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    output = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = ABNORMAL_COLOR if det.is_abnormal else HEALTHY_COLOR
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(output, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return output


def analyze_image(
    image_input,  # np.ndarray or PIL.Image or file path
    model=None,
    conf_threshold: float = 0.3,
    use_opencv_fallback: bool = False
) -> Tuple[np.ndarray, List[CellResult], dict]:
    """
    Main entry point. Returns annotated image, list of detections, and summary stats.
    """
    # Normalize input to np.ndarray BGR
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        image = image_input

    if image is None:
        raise ValueError("Could not load image")

    # Run detection
    if use_opencv_fallback or model is None:
        detections = detect_with_opencv_fallback(image)
    else:
        detections = detect_with_yolo(image, model, conf_threshold)

    # Draw results
    annotated = draw_detections(image, detections)

    # Build summary
    class_counts = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

    abnormal_count = sum(1 for d in detections if d.is_abnormal)
    healthy_count = len(detections) - abnormal_count
    avg_conf = np.mean([d.confidence for d in detections]) if detections else 0.0

    summary = {
        "total_cells": len(detections),
        "healthy_cells": healthy_count,
        "abnormal_cells": abnormal_count,
        "class_counts": class_counts,
        "confidence_avg": round(float(avg_conf), 3),
        "detection_method": "OpenCV Fallback" if (use_opencv_fallback or model is None) else "YOLOv8",
    }

    return annotated, detections, summary
