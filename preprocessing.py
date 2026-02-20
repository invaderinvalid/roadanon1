"""Preprocessing: threaded capture + resize for all sources.

Thread layout (RPi 4B — 4 cores):
  Core 1: Camera/file capture thread (I/O + decode)
  Core 2: Main thread (motion + tracker + draw)
  Core 3-4: YOLO thread (NCNN inference)
"""

import cv2
import numpy as np
from threading import Thread, Lock
from collections import deque
from config import Config


class ThreadedVideoSource:
    """Threaded capture for live cameras — always returns the LATEST frame.

    Drops intermediate frames to stay real-time (no queue buildup).
    """

    def __init__(self, source, width, height):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self._width = width
        self._height = height
        self._lock = Lock()
        self._frame = None
        self._stopped = False
        ret, frame = self._cap.read()
        if ret:
            self._frame = self._resize(frame)
        else:
            self._stopped = True
        self._thread = Thread(target=self._update, daemon=True)
        self._thread.start()

    def _resize(self, frame):
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            return cv2.resize(frame, (self._width, self._height))
        return frame

    def _update(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                self._stopped = True
                return
            resized = self._resize(frame)
            with self._lock:
                self._frame = resized

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        if self._stopped:
            return None
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2)
        self._cap.release()


class ThreadedFileSource:
    """Threaded capture for video files — read-ahead buffer.

    Decodes frames on a background thread into a small ring buffer.
    Main thread picks up pre-decoded, pre-resized frames instantly.
    Unlike ThreadedVideoSource, this preserves every frame (no drops).
    """

    def __init__(self, source, width, height, buffer_size=8):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self._width = width
        self._height = height
        self._lock = Lock()
        self._buffer = deque(maxlen=buffer_size)
        self._stopped = False
        self._exhausted = False
        self._thread = Thread(target=self._update, daemon=True)
        self._thread.start()

    def _resize(self, frame):
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            return cv2.resize(frame, (self._width, self._height))
        return frame

    def _update(self):
        while not self._stopped:
            # Don't overfill buffer — wait if full
            with self._lock:
                buf_len = len(self._buffer)
            if buf_len >= self._buffer.maxlen:
                import time
                time.sleep(0.001)
                continue

            ret, frame = self._cap.read()
            if not ret:
                self._exhausted = True
                return
            resized = self._resize(frame)
            with self._lock:
                self._buffer.append(resized)

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        import time
        # Wait briefly if buffer empty but source still has frames
        for _ in range(200):  # max ~200ms wait
            with self._lock:
                if self._buffer:
                    return self._buffer.popleft()
            if self._exhausted:
                return None
            time.sleep(0.001)
        return None  # timeout

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2)
        self._cap.release()


class MultiCamera:
    """Manages multiple video sources with threaded I/O."""

    def __init__(self, cfg: Config):
        self.sources = []
        for s in cfg.sources:
            if isinstance(s, int):
                # Live camera — drop-to-latest
                self.sources.append(
                    ThreadedVideoSource(s, cfg.proc_width, cfg.proc_height))
            else:
                # Video file — read-ahead buffer
                self.sources.append(
                    ThreadedFileSource(s, cfg.proc_width, cfg.proc_height))
        self.width = cfg.proc_width
        self.height = cfg.proc_height
        print(f"[Camera] {len(self.sources)} source(s), threaded I/O, "
              f"{cfg.proc_width}x{cfg.proc_height}")

    @property
    def src_fps(self):
        return self.sources[0].fps if self.sources else 30.0

    def read_all(self):
        """Return list of pre-resized frames (None if exhausted)."""
        return [src.read() for src in self.sources]

    def release(self):
        for s in self.sources:
            s.release()
