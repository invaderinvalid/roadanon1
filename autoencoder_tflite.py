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
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            # Detect expected dtype (fp16 models need float16 input)
            self._input_fp16 = (inp.type == "tensor(float16)")
            # Use model's actual input shape (overrides config if different)
            model_shape = inp.shape  # e.g. [1, 3, 128, 128]
            if len(model_shape) == 4 and isinstance(model_shape[-1], int):
                self.input_size = model_shape[-1]
            print(f"[AE] Loaded {model_path} (input: {self.input_size}x{self.input_size}, "
                  f"{'fp16' if self._input_fp16 else 'fp32'})")
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

        # Cast to model's expected dtype
        if self._input_fp16:
            blob = blob.astype(np.float16)

        output = self.session.run(None, {self.input_name: blob})[0]
        # Compute error in float32 for stability
        blob_f32 = blob.astype(np.float32) if self._input_fp16 else blob
        out_f32 = output.astype(np.float32) if output.dtype != np.float32 else output
        error = np.abs(blob_f32 - out_f32).mean(axis=1).squeeze(0)
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
