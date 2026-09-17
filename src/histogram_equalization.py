"""
histogram_equalization.py
-------------------------
Global and CLAHE contrast enhancement.
"""
import cv2
import numpy as np


def global_equalize(image: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization to a grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.equalizeHist(gray)


def clahe_equalize(image: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        clip: Threshold for contrast limiting.
        grid: Size of the tile grid for local equalization.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)
