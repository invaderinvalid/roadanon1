"""Motion detection using background subtraction."""

import cv2
import numpy as np
from config import Config


class MotionDetector:
    """Detects moving objects and returns binary masks + regions."""

    def __init__(self, cfg: Config):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=cfg.motion_history, varThreshold=cfg.motion_threshold,
            detectShadows=False)
        self.min_area = cfg.min_contour_area
        self.erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame: np.ndarray):
        """
        Returns:
            mask: binary motion mask
            regions: list of ((x, y, w, h), area)
        """
        mask = self.mog2.apply(frame)
        mask = cv2.erode(mask, self.erode_k, iterations=1)
        mask = cv2.dilate(mask, self.dilate_k, iterations=2)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_area:
                regions.append((cv2.boundingRect(c), area))
        return mask, regions
