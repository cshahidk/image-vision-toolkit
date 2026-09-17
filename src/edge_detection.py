"""
edge_detection.py
-----------------
Modern edge detection using Canny and Sobel operators.
Author: Shahid Karim
License: MIT
"""
import cv2
import numpy as np


def canny_edges(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Detect edges using the Canny algorithm.

    Args:
        image: Input BGR or grayscale image (uint8).
        low:   Lower hysteresis threshold.
        high:  Upper hysteresis threshold.

    Returns:
        Binary edge map (uint8, values 0 or 255).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    return cv2.Canny(blurred, low, high)


def sobel_gradient(image: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Compute Sobel gradients in x and y directions.

    Returns:
        (grad_x, grad_y) as float32 arrays.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return grad_x, grad_y


def gradient_magnitude(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """Return the L2 magnitude of the gradient, normalized to [0, 255]."""
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)
