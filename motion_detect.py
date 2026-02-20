"""Motion detection — fast frame differencing for RPi.

Replaces OpenCV MOG2 which is too slow on ARM Cortex-A72.
Frame differencing: ~2-5ms vs MOG2: ~200ms on RPi 4B.
"""

import cv2
import numpy as np
from config import Config


class MotionDetector:
    """Fast motion detection using frame differencing.

    Algorithm:
      1. Convert to grayscale
      2. Gaussian blur (reduce noise)
      3. absdiff(current, previous)
      4. Threshold → contours
    """

    def __init__(self, cfg: Config):
        self.min_area = cfg.min_contour_area
        self._prev_gray = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self._threshold = int(cfg.motion_threshold)

    def detect(self, frame: np.ndarray):
        """
        Returns:
            mask: binary motion mask (or None on first frame)
            regions: list of ((x, y, w, h), area)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None, []

        # Frame difference
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        # Threshold + clean up
        _, mask = cv2.threshold(diff, self._threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_area:
                regions.append((cv2.boundingRect(c), area))

        return mask, regions
