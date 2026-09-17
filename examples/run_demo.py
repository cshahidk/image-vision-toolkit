"""
run_demo.py
-----------
Quick demo: run all four utilities on a sample image.
Usage:
    python examples/run_demo.py data/sample.jpg
"""
import sys
import cv2
from src.edge_detection import canny_edges, sobel_gradient, gradient_magnitude
from src.histogram_equalization import clahe_equalize
from src.image_denoising import bilateral_denoise
from src.feature_matching import detect_and_match


def main(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")

    edges = canny_edges(img)
    gx, gy = sobel_gradient(img)
    mag = gradient_magnitude(gx, gy)
    enhanced = clahe_equalize(img)
    denoised = bilateral_denoise(img)

    cv2.imwrite("out_canny.png", edges)
    cv2.imwrite("out_sobel.png", mag)
    cv2.imwrite("out_clahe.png", enhanced)
    cv2.imwrite("out_denoised.png", denoised)

    # Self-matching sanity check
    matches = detect_and_match(img, img)
    print(f"ORB self-matches: {len(matches)}")
    print("Outputs written: out_canny.png, out_sobel.png, out_clahe.png, out_denoised.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample.jpg")
