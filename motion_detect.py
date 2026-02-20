"""Motion detection using background subtraction — optimized for RPi."""

import cv2
import numpy as np
from config import Config


class MotionDetector:
    """Detects moving objects using lightweight background subtraction."""

    def __init__(self, cfg: Config):
        # MOG2 with reduced history and fewer gaussians for ARM speed
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=cfg.motion_history,
            varThreshold=cfg.motion_threshold,
            detectShadows=False,
        )
        # Reduce number of gaussian mixtures (default=5, we use 3)
        self.mog2.setNMixtures(3)
        self.min_area = cfg.min_contour_area

        # Single combined kernel instead of separate erode+dilate
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Downscale factor for motion mask computation
        self._scale = 0.5
        self._small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def detect(self, frame: np.ndarray):
        """
        Returns:
            mask: binary motion mask (original resolution)
            regions: list of ((x, y, w, h), area) in original coords
        """
        h, w = frame.shape[:2]

        # Downscale for faster BG subtraction
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

        # Apply MOG2 on smaller frame
        mask = self.mog2.apply(small, learningRate=0.005)

        # Single morphological close (fills gaps) instead of erode+dilate
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._small_kernel)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # Find contours on small mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Scale bounding boxes back to original resolution
        regions = []
        for c in contours:
            area = cv2.contourArea(c) * 4  # compensate for 0.5x scale
            if area >= self.min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                regions.append(((x * 2, y * 2, bw * 2, bh * 2), area))

        return mask, regions
