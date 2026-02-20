"""Motion detection using background subtraction — balanced for RPi."""

import cv2
import numpy as np
from config import Config


class MotionDetector:
    """Detects moving objects using background subtraction.

    No downscaling — the ROI is already small (320x156 on RPi preset).
    """

    def __init__(self, cfg: Config):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=cfg.motion_history,
            varThreshold=cfg.motion_threshold,
            detectShadows=False,
        )
        self.mog2.setNMixtures(3)
        self.min_area = cfg.min_contour_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray):
        """
        Returns:
            mask: binary motion mask
            regions: list of ((x, y, w, h), area)
        """
        mask = self.mog2.apply(frame, learningRate=0.005)

        # Single close operation — fills small gaps without heavy cost
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_area:
                regions.append((cv2.boundingRect(c), area))

        return mask, regions
