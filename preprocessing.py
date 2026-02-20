"""Preprocessing: capture frames and resize.

Includes threaded capture for RPi to hide I/O latency.
"""

import cv2
import numpy as np
from threading import Thread, Lock
from config import Config


class VideoSource:
    """Manages a single video source (camera or file)."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


class ThreadedVideoSource:
    """Threaded wrapper — reads frames in background to hide I/O latency."""

    def __init__(self, source):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self._lock = Lock()
        self._frame = None
        self._stopped = False
        ret, self._frame = self._cap.read()
        if not ret:
            self._stopped = True
        self._thread = Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                self._stopped = True
                return
            with self._lock:
                self._frame = frame

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self):
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self):
        if self._stopped:
            return None
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2)
        self._cap.release()


def preprocess(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to (width, height)."""
    if frame.shape[1] != width or frame.shape[0] != height:
        return cv2.resize(frame, (width, height))
    return frame


class MultiCamera:
    """Manages multiple video sources."""

    def __init__(self, cfg: Config):
        # Use threaded capture for live cameras (int source), sync for files
        self.sources = []
        for s in cfg.sources:
            if isinstance(s, int):
                self.sources.append(ThreadedVideoSource(s))
            else:
                self.sources.append(VideoSource(s))
        self.width = cfg.proc_width
        self.height = cfg.proc_height

    @property
    def src_fps(self):
        """FPS of first source (for display throttling)."""
        return self.sources[0].fps if self.sources else 30.0

    def read_all(self):
        """Return list of preprocessed frames (None if source exhausted)."""
        frames = []
        for src in self.sources:
            f = src.read()
            if f is not None:
                f = preprocess(f, self.width, self.height)
            frames.append(f)
        return frames

    def release(self):
        for s in self.sources:
            s.release()
