"""
scripts/benchmark.py - Measure real end-to-end processing time for analyze_image(),
comparing full-resolution input vs. resizing large images down to a max
dimension before detection.

Run this yourself and use the number it prints - don't guess one.

Usage:
    python scripts/benchmark.py --image samples/BloodImage_00010.jpg --runs 10
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cell_detector import analyze_image, load_model


def resize_if_large(image: np.ndarray, max_dim: int = 960) -> np.ndarray:
    """Downscale an image so its longer side is at most max_dim, preserving aspect ratio."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def time_runs(image: np.ndarray, model, runs: int) -> float:
    # one untimed warmup run - model/JIT warmup shouldn't count against either condition
    analyze_image(image, model=model, use_opencv_fallback=(model is None))

    start = time.perf_counter()
    for _ in range(runs):
        analyze_image(image, model=model, use_opencv_fallback=(model is None))
    elapsed = time.perf_counter() - start
    return elapsed / runs


def main():
    parser = argparse.ArgumentParser(description="Benchmark analyze_image() with and without pre-resize")
    parser.add_argument("--image", required=True, help="Path to a test image (e.g. a file in samples/)")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed runs to average")
    parser.add_argument("--max-dim", type=int, default=960, help="Max longer-side dimension for the resized condition")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    original = cv2.imread(args.image)
    if original is None:
        print(f"Could not read image: {args.image}")
        sys.exit(1)

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.pt")
    model = load_model(model_path if os.path.exists(model_path) else None)

    resized = resize_if_large(original, args.max_dim)

    print(f"Original size: {original.shape[1]}x{original.shape[0]}")
    print(f"Resized size:  {resized.shape[1]}x{resized.shape[0]}")
    print(f"Averaging over {args.runs} timed runs (after 1 untimed warmup run)...\n")

    baseline = time_runs(original, model, args.runs)
    optimized = time_runs(resized, model, args.runs)

    pct_change = (baseline - optimized) / baseline * 100 if baseline > 0 else 0.0

    print(f"Baseline (full resolution):  {baseline * 1000:.1f} ms/run")
    print(f"Optimized (resized to {args.max_dim}px): {optimized * 1000:.1f} ms/run")
    print(f"Change: {pct_change:.1f}% {'faster' if pct_change > 0 else 'slower'}")
    print("\nThis number is specific to this image, this machine, and this run count.")
    print("Run it across a few different sample images before quoting a single figure.")


if __name__ == "__main__":
    main()
