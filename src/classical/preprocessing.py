"""
src/classical/preprocessing.py

Week 1-2 of the course implemented:
    - Grayscale conversion
    - CLAHE (contrast limited adaptive histogram equalization)
    - Gaussian denoising
    - Canny edge detection
    - Visualization utilities

This is the first stage of the project pipeline.
All operations use OpenCV and run before any deep learning.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from configs.config import (
    GAUSSIAN_KERNEL, GAUSSIAN_SIGMA,
    CLAHE_CLIP, CLAHE_GRID,
    CANNY_LOW, CANNY_HIGH,
    FIGURES_DIR
)


# ─── Individual Operations ────────────────────────────────────────────────────

def load_image(path, grayscale=True):
    """Load image from path. Returns numpy array (H, W) for grayscale."""
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def apply_clahe(image):
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization.
    Unlike global histogram equalization, CLAHE operates on small tiles
    and limits contrast amplification to avoid noise amplification.
    Essential for chest X-rays where local contrast matters more than global.
    Input:  grayscale uint8 image
    Output: contrast-enhanced grayscale image
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(image)


def apply_gaussian_denoise(image):
    """
    Gaussian smoothing to suppress noise before edge detection.
    Kernel size and sigma set in config.
    Input:  grayscale image
    Output: smoothed image
    """
    return cv2.GaussianBlur(image, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)


def apply_canny(image):
    """
    Canny edge detection.
    Automatically handles: Gaussian smoothing → gradient → non-max suppression
    → hysteresis thresholding.
    Input:  grayscale image (ideally already smoothed)
    Output: binary edge map (0 or 255)
    """
    return cv2.Canny(image, CANNY_LOW, CANNY_HIGH)


def apply_sobel(image):
    """
    Sobel edge detection — returns gradient magnitude map.
    Useful for comparing classical vs Canny approach.
    Input:  grayscale image
    Output: gradient magnitude image (float32, normalized to 0-255)
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)   # horizontal edges
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)   # vertical edges
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude


def normalize_image(image):
    """
    Normalize pixel values to [0, 1] float32.
    Required before feeding into neural networks.
    """
    return image.astype(np.float32) / 255.0


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def preprocess_xray(image_path):
    """
    Full classical preprocessing pipeline for one chest X-ray.
    Returns a dict of all intermediate results for visualization.

    Pipeline:
        Raw → Grayscale → CLAHE → Gaussian Denoise → Canny Edges
    """
    raw         = load_image(image_path, grayscale=True)
    clahe       = apply_clahe(raw)
    denoised    = apply_gaussian_denoise(clahe)
    edges_canny = apply_canny(denoised)
    edges_sobel = apply_sobel(denoised)
    normalized  = normalize_image(denoised)

    return {
        "raw":          raw,
        "clahe":        clahe,
        "denoised":     denoised,
        "edges_canny":  edges_canny,
        "edges_sobel":  edges_sobel,
        "normalized":   normalized,
    }


def preprocess_batch(image_paths):
    """
    Run full preprocessing on a list of image paths.
    Returns list of normalized numpy arrays ready for model input.
    """
    processed = []
    for path in image_paths:
        result = preprocess_xray(path)
        processed.append(result["normalized"])
    return np.array(processed)


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_pipeline(image_path, save=False):
    """
    Plot all stages of the preprocessing pipeline side by side.
    Useful for understanding what each step does to the X-ray.
    """
    results = preprocess_xray(image_path)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ["Raw", "CLAHE", "Denoised", "Canny Edges", "Sobel Edges"]
    keys   = ["raw", "clahe", "denoised", "edges_canny", "edges_sobel"]

    for ax, title, key in zip(axes, titles, keys):
        ax.imshow(results[key], cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.suptitle("Classical Preprocessing Pipeline — Chest X-ray", fontsize=14)
    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(FIGURES_DIR, "classical_pipeline.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()

    return results


if __name__ == "__main__":
    # Quick test with a dummy image
    import tempfile
    from PIL import Image as PILImage

    dummy_path = "/tmp/test_xray.png"
    arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
    PILImage.fromarray(arr).save(dummy_path)

    results = preprocess_xray(dummy_path)
    print("Pipeline ran successfully.")
    for k, v in results.items():
        print(f"  {k:15s}: shape={v.shape}, dtype={v.dtype}")
