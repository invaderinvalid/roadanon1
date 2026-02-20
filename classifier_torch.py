"""PyTorch YOLO classifier for Mac/dev testing. Same interface as classifier_ncnn."""

import numpy as np
import cv2
from config import Config

CLASSES = ["road_damage", "speed_bump", "unsurfaced_road"]


class YOLOClassifierTorch:
    """Loads a YOLOv26 best.pt via ultralytics — returns same dict format as NCNN."""

    def __init__(self, cfg: Config):
        self.conf_threshold = cfg.yolo_conf
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(cfg.yolo_pt_path)
            print(f"[YOLO-Torch] Loaded {cfg.yolo_pt_path}")
        except Exception as e:
            print(f"[YOLO-Torch] WARNING: {e}")

    def infer(self, frame):
        """Returns list of dicts: {bbox, class_id, class_name, confidence}."""
        if self.model is None:
            return []

        results = self.model(frame, verbose=False)[0]
        detections = []

        if hasattr(results, "boxes") and len(results.boxes):
            for box in results.boxes:
                conf = float(box.conf.cpu())
                if conf < self.conf_threshold:
                    continue
                cls_id = int(box.cls.cpu())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}"
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": round(conf, 3),
                })

        return detections
