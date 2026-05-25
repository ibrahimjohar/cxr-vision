"""
Classical image preprocessing pipeline for chest X-rays.
Includes:
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
from configs.config import (GAUSSIAN_KERNEL, GAUSSIAN_SIGMA, CLAHE_CLIP, CLAHE_GRID, CANNY_LOW, CANNY_HIGH, FIGURES_DIR, DATA_RAW)

#individual operations
def load_image(path, grayscale=True):
    #load img from path - returns numpy array (H,W) for grayscale
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

#CLAHE - contrast limited adaptive histogram equalization
#unlike global histogram equalization, CLAHE operates on small tiles and limits contrast amplification to avoid noise amplification
#this is essential for chest X-rays, where local contrast matters more than global
#input: grayscale uint8 img
#output: contrast-enhanced grayscale img

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(image)

#gaussian smoothing to suppress noise before edge detection
#kernel size & sigma set in config
#input: grayscale img
#output: smoothed img

def apply_gaussian_denoise(image):
    return cv2.GaussianBlur(image, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)

#canny edge detection
#automatically handles: gaussian smoothing -> gradient -> non-max suppression -> hysteresis thresholding
#input: grayscale img
#output: binary edge map (0 or 255)

def apply_canny(image):
    return cv2.Canny(image, CANNY_LOW, CANNY_HIGH)

#sobel edge detection - returns gradient magnitude map
#useful for our comparision of classical vs Canny approach
#input: grayscale img
#output: gradient magnitude map (float32, normalised to 0-255)

def apply_sobel(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) #horizontal edges
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) #vertical edges
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude

#normalize pixel vals to [0,1] float32
#this is required before feeding into neural networks

def normalize_image(image):
    return image.astype(np.float32) / 255.0



#full classical preprocessing pipeline for one chest x-ray image
#returns a dict of all intermediate results for visualization

#pipeline: raw -> grayscale -> CLAHE -> gaussian denoise -> canny edges

def preprocess_xray(image_path):
    raw = load_image(image_path, grayscale=True)
    clahe = apply_clahe(raw)
    denoised = apply_gaussian_denoise(clahe)
    edges_canny = apply_canny(denoised) 
    edges_sobel = apply_sobel(denoised)
    normalised = normalize_image(denoised)
    
    return {
        "raw": raw,
        "clahe": clahe,
        "denoised": denoised,
        "edges_canny": edges_canny,
        "edges_sobel": edges_sobel,
        "normalised": normalised
    }
    

#run full preprocessing on a list of img paths
#returns list of normalized numpy arrays ready for model input

def preprocess_batch(image_paths):
    processed = []
    for path in image_paths:
        result = preprocess_xray(path)
        processed.append(result["normalised"])
    return np.array(processed)


def visualize_pipeline(image_path, save=False):
    """
    plot all stages of the preprocessing pipeline side by side.
    useful for understanding what each step does to the X-ray.
    """
    results = preprocess_xray(image_path)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ["Raw", "CLAHE", "Denoised", "Canny Edges", "Sobel Edges"]
    keys = ["raw", "clahe", "denoised", "edges_canny", "edges_sobel"]

    for ax, title, key in zip(axes, titles, keys):
        ax.imshow(results[key], cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.suptitle("Classical Preprocessing Pipeline - Chest X-ray", fontsize=14)
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

    dummy_path = os.path.join(DATA_RAW, "dummy", "test_xray.png")
    os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
    arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
    PILImage.fromarray(arr).save(dummy_path)

    results = preprocess_xray(dummy_path)
    print("Pipeline ran successfully.")
    for k, v in results.items():
        print(f"  {k:15s}: shape={v.shape}, dtype={v.dtype}")  
    
    
    

     