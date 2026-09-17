"""
image_denoising.py
------------------
Denoising utilities: Gaussian, median, bilateral, and non-local means.
"""
import cv2
import numpy as np


def gaussian_denoise(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def median_denoise(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Excellent for salt-and-pepper noise."""
    return cv2.medianBlur(image, ksize)


def bilateral_denoise(image: np.ndarray, d: int = 9,
                      sigma_color: float = 75,
                      sigma_space: float = 75) -> np.ndarray:
    """Edge-preserving denoising."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def nlm_denoise(image: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Non-Local Means denoising (slow but high quality)."""
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
