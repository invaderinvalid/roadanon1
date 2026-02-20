"""Cropping ROIs and filtering black/empty images."""

import cv2
import numpy as np
from config import Config


class Cropper:
    """Extracts bounding-box crops from high-error regions."""

    def __init__(self, cfg: Config):
        self.padding = cfg.crop_padding
        self.min_size = cfg.crop_min_size

    def extract(self, frame, error_map, threshold):
        """Returns list of dicts: {bbox, image, score}."""
        binary = (error_map > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.dilate(binary, kernel, iterations=2)
        binary = cv2.erode(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        crops = []

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < self.min_size or bh < self.min_size:
                continue
            x1 = max(0, x - self.padding)
            y1 = max(0, y - self.padding)
            x2 = min(w, x + bw + self.padding)
            y2 = min(h, y + bh + self.padding)
            crops.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "image": frame[y1:y2, x1:x2].copy(),
                "score": float(error_map[y:y+bh, x:x+bw].mean()),
            })

        crops.sort(key=lambda c: c["score"], reverse=True)
        return crops


class Validator:
    """Filters out mostly-black crops (masking artifacts)."""

    def __init__(self, cfg: Config):
        self.max_black_ratio = cfg.black_max_ratio
        self.black_thresh = cfg.black_pixel_thresh

    def is_valid(self, crop_img):
        if crop_img.size == 0 or crop_img.shape[0] < 8 or crop_img.shape[1] < 8:
            return False
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        black = np.sum(gray < self.black_thresh)
        return (black / gray.size) < self.max_black_ratio
