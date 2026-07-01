    """
tests/test_detector.py - Unit tests for cell detection pipeline
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cell_detector import (
    detect_with_opencv_fallback,
    draw_detections,
    analyze_image,
    CellResult
)


def make_synthetic_cell_image(n_cells: int = 5, size: int = 400) -> np.ndarray:
    """Create a synthetic image with circles representing cells."""
    image = np.ones((size, size, 3), dtype=np.uint8) * 200  # Light gray background
    np.random.seed(42)
    for _ in range(n_cells):
        cx = np.random.randint(50, size - 50)
        cy = np.random.randint(50, size - 50)
        radius = np.random.randint(15, 35)
        color = (np.random.randint(100, 200), np.random.randint(50, 150), np.random.randint(50, 150))
        cv2.circle(image, (cx, cy), radius, color, -1)
        cv2.circle(image, (cx, cy), radius, (0, 0, 0), 1)
    return image


class TestOpenCVDetector:
    def test_detects_cells_in_synthetic_image(self):
        image = make_synthetic_cell_image(n_cells=5)
        detections = detect_with_opencv_fallback(image)
        # Should detect at least some cells
        assert len(detections) > 0

    def test_returns_cell_result_objects(self):
        image = make_synthetic_cell_image(n_cells=3)
        detections = detect_with_opencv_fallback(image)
        for d in detections:
            assert isinstance(d, CellResult)
            assert d.class_name in ["RBC", "WBC", "Platelet"]
            assert 0.0 <= d.confidence <= 1.0
            assert len(d.bbox) == 4
            assert d.area > 0

    def test_empty_image_returns_no_detections(self):
        # Pure white image with no cells
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        detections = detect_with_opencv_fallback(image)
        assert len(detections) == 0

    def test_bbox_within_image_bounds(self):
        image = make_synthetic_cell_image()
        h, w = image.shape[:2]
        detections = detect_with_opencv_fallback(image)
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            assert 0 <= x1 < w
            assert 0 <= y1 < h
            assert x1 < x2
            assert y1 < y2


class TestDrawDetections:
    def test_draw_returns_same_shape(self):
        image = make_synthetic_cell_image()
        h, w = image.shape[:2]
        detections = detect_with_opencv_fallback(image)
        output = draw_detections(image, detections)
        assert output.shape == (h, w, 3)

    def test_draw_does_not_modify_original(self):
        image = make_synthetic_cell_image()
        original = image.copy()
        detections = detect_with_opencv_fallback(image)
        draw_detections(image, detections)
        np.testing.assert_array_equal(image, original)


class TestAnalyzeImage:
    def test_returns_correct_tuple(self):
        image = make_synthetic_cell_image()
        annotated, detections, summary = analyze_image(image, model=None, use_opencv_fallback=True)
        assert isinstance(annotated, np.ndarray)
        assert isinstance(detections, list)
        assert isinstance(summary, dict)

    def test_summary_has_required_keys(self):
        image = make_synthetic_cell_image()
        _, _, summary = analyze_image(image, model=None, use_opencv_fallback=True)
        required_keys = ["total_cells", "healthy_cells", "abnormal_cells", "confidence_avg", "detection_method"]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_cell_counts_are_consistent(self):
        image = make_synthetic_cell_image()
        _, detections, summary = analyze_image(image, model=None, use_opencv_fallback=True)
        assert summary["total_cells"] == len(detections)
        assert summary["healthy_cells"] + summary["abnormal_cells"] == summary["total_cells"]

    def test_accepts_pil_image(self):
        from PIL import Image
        image = make_synthetic_cell_image()
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        _, _, summary = analyze_image(pil_image, model=None, use_opencv_fallback=True)
        assert "total_cells" in summary
