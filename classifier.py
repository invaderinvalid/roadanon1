"""YOLOv26n TFLite classifier for anomaly classification."""

import numpy as np
import cv2
from config import Config

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback for dev machines
    import tensorflow.lite as tflite
    Interpreter = tflite.Interpreter


CLASSES = ["road_damage", "speed_bump", "unsurfaced_road"]


class YOLOClassifier:
    """Runs quantized YOLOv26n TFLite model on cropped regions."""

    def __init__(self, cfg: Config):
        self.input_size = cfg.yolo_input_size
        self.conf_threshold = cfg.yolo_conf
        self.interpreter = None

        try:
            self.interpreter = Interpreter(model_path=cfg.yolo_model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[YOLO] Loaded {cfg.yolo_model_path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[YOLO] WARNING: Could not load model: {e}")

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize and normalize crop for TFLite input."""
        img = cv2.resize(crop, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = self.input_details[0]
        if inp["dtype"] == np.uint8:
            return np.expand_dims(img, axis=0).astype(np.uint8)
        return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    def classify(self, crop: np.ndarray) -> tuple:
        """
        Classify a single crop.

        Returns:
            (label: str, confidence: float) or ("unknown_anomaly", 0.0) if no model
        """
        if self.interpreter is None:
            return "unknown_anomaly", 0.0

        input_data = self._preprocess(crop)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        scores = output.flatten()

        # Map output to classes
        if len(scores) >= len(CLASSES):
            idx = int(np.argmax(scores[:len(CLASSES)]))
            conf = float(scores[idx])
            if conf >= self.conf_threshold:
                return CLASSES[idx], conf
        return "unknown_anomaly", float(scores.max()) if len(scores) > 0 else 0.0

    def classify_batch(self, crops: list) -> list:
        """Classify multiple crops."""
        return [self.classify(c) for c in crops]
