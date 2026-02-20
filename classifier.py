"""YOLOv26n TFLite classifier for anomaly classification."""

import sys
import numpy as np
import cv2
from config import Config

# Python 3.12 removed the 'imp' module, but old tflite_runtime still imports it.
# Provide a shim so tflite_runtime can load.
if sys.version_info >= (3, 12) and 'imp' not in sys.modules:
    import importlib
    sys.modules['imp'] = importlib

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        Interpreter = None


CLASSES = ["road_damage", "speed_bump", "unsurfaced_road"]


class YOLOClassifier:
    """Runs quantized YOLOv26n TFLite model on cropped regions."""

    def __init__(self, cfg: Config):
        self.input_size = cfg.yolo_input_size
        self.conf_threshold = cfg.yolo_conf
        self.interpreter = None

        if Interpreter is None:
            print("[YOLO-TFLite] WARNING: tflite_runtime not installed")
            return

        try:
            self.interpreter = Interpreter(model_path=cfg.yolo_tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[YOLO-TFLite] Loaded {cfg.yolo_tflite_path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[YOLO-TFLite] WARNING: Could not load model: {e}")

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

    def infer(self, frame):
        """Run detection on a full frame — same interface as NCNN/Torch classifiers.

        Returns list of dicts: {bbox, class_id, class_name, confidence}
        bbox is (x, y, w, h) in original frame coords.
        """
        if self.interpreter is None:
            return []

        input_data = self._preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Handle YOLOv26 output: [1, 4+nc, num_anchors] or [1, num_anchors, 4+nc]
        out = output.squeeze()
        if out.ndim != 2:
            # Fallback: treat as classification on whole frame
            scores = out.flatten()
            if len(scores) >= len(CLASSES):
                idx = int(np.argmax(scores[:len(CLASSES)]))
                conf = float(scores[idx])
                if conf >= self.conf_threshold:
                    h, w = frame.shape[:2]
                    return [{"bbox": (0, 0, w, h), "class_id": idx,
                             "class_name": CLASSES[idx], "confidence": round(conf, 3)}]
            return []

        # Determine orientation: [4+nc, anchors] or [anchors, 4+nc]
        if out.shape[0] == 4 + len(CLASSES):
            # [4+nc, anchors] format
            num_anchors = out.shape[1]
        else:
            # [anchors, 4+nc] format — transpose
            out = out.T
            num_anchors = out.shape[1]

        h, w = frame.shape[:2]
        scale_x = w / self.input_size
        scale_y = h / self.input_size
        boxes, scores, class_ids = [], [], []

        for i in range(num_anchors):
            cls_scores = out[4:, i]
            class_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[class_id])
            if conf < self.conf_threshold:
                continue
            cx, cy, bw, bh = out[0, i], out[1, i], out[2, i], out[3, i]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            bw_out = int(bw * scale_x)
            bh_out = int(bh * scale_y)
            boxes.append([x1, y1, bw_out, bh_out])
            scores.append(conf)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, 0.45)
        detections = []
        for idx in indices:
            i = int(idx)
            x, y, bw, bh = boxes[i]
            cid = class_ids[i]
            name = CLASSES[cid] if cid < len(CLASSES) else f"cls{cid}"
            detections.append({
                "bbox": (x, y, bw, bh),
                "class_id": cid,
                "class_name": name,
                "confidence": round(scores[i], 3),
            })
        return detections
