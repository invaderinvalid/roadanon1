"""Autoencoder anomaly detection — ONNX Runtime FP16 for RPi.

Optimizations:
  - Configurable input size (64x64 default → <10ms on RPi vs 40ms at 128x128)
  - Temporal smoothing: rolling window of errors to filter spikes
"""

import os
import numpy as np
import cv2
from collections import deque
from config import Config

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False


class AutoencoderDetector:
    """ONNX Runtime FP16 autoencoder with temporal smoothing."""

    def __init__(self, cfg: Config):
        self.threshold = cfg.ae_threshold
        self.input_size = cfg.ae_input_size
        self.session = None

        # Temporal smoothing — require N of last M frames anomalous
        self._error_window = deque(maxlen=cfg.ae_smooth_window)
        self._smooth_min_hits = cfg.ae_smooth_min_hits

        if not _HAS_ORT:
            print("[AE] WARNING: onnxruntime not installed")
            return

        model_path = cfg.ae_model_path
        if not os.path.exists(model_path):
            print(f"[AE] WARNING: {model_path} not found — run convert_ae_tflite.py")
            return

        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                model_path, opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            print(f"[AE] Loaded {model_path} (input: {self.input_size}x{self.input_size})")
        except Exception as e:
            self.session = None
            print(f"[AE] WARNING: {e}")

    def infer(self, frame: np.ndarray):
        """
        Run autoencoder.

        Returns:
            error_map: per-pixel reconstruction error (resized to frame dims)
            mean_error: float
        """
        if self.session is None:
            return None, 0.0

        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

        output = self.session.run(None, {self.input_name: blob})[0]
        error = np.abs(blob - output).mean(axis=1).squeeze(0)
        mean_error = float(error.mean())
        error_map = cv2.resize(error, (w, h))

        return error_map, mean_error

    def is_anomaly_smoothed(self, frame: np.ndarray, effective_threshold: float):
        """Check for anomaly with temporal smoothing.

        Args:
            frame: input frame
            effective_threshold: motion-weighted threshold

        Returns:
            (is_anomaly: bool, mean_error: float, smoothed_count: int)
        """
        _, mean_error = self.infer(frame)

        # Add to rolling window
        self._error_window.append(mean_error)

        # Count how many recent frames exceeded threshold
        hits = sum(1 for e in self._error_window if e > effective_threshold)

        # Require persistence — N of last M frames anomalous
        is_anomaly = hits >= self._smooth_min_hits

        return is_anomaly, mean_error, hits
