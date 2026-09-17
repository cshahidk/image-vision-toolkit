"""
feature_matching.py
-------------------
ORB feature detection and brute-force matching between two images.
"""
import cv2
import numpy as np


def detect_and_match(img1: np.ndarray, img2: np.ndarray,
                     max_features: int = 1000,
                     ratio_thresh: float = 0.75) -> list[cv2.DMatch]:
    """Detect ORB features and match with Lowe's ratio test.

    Args:
        img1, img2: Input images (BGR or grayscale).
        max_features: Maximum ORB keypoints per image.
        ratio_thresh: Lowe's ratio threshold (0.7–0.8 typical).

    Returns:
        List of good cv2.DMatch objects.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return good
