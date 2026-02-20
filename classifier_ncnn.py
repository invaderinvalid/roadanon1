"""NCNN YOLOv26n classifier for road anomaly detection."""

import numpy as np
import cv2
from config import Config

try:
    import ncnn
except ImportError:
    ncnn = None

CLASSES = ["road_damage", "speed_bump", "unsurfaced_road"]


class YOLOClassifier:
    """
    NCNN YOLOv26n inference.
    Input:  in0  — [3, 640, 640]
    Output: out0 — [4+nc, 8400]  (cx, cy, w, h, cls_scores...)
    """

    def __init__(self, cfg: Config):
        self.input_size = cfg.yolo_input_size
        self.conf_threshold = cfg.yolo_conf
        self.nms_threshold = cfg.yolo_nms
        self.num_classes = len(CLASSES)
        self.net = None

        if ncnn is None:
            print("[YOLO] WARNING: ncnn not installed — pip install ncnn")
            return
        try:
            self.net = ncnn.Net()
            self.net.opt.use_vulkan_compute = False
            self.net.opt.num_threads = 1
            self.net.load_param(cfg.yolo_ncnn_param)
            self.net.load_model(cfg.yolo_ncnn_bin)
            print(f"[YOLO] Loaded NCNN model ({self.num_classes} classes)")
        except Exception as e:
            self.net = None
            print(f"[YOLO] WARNING: {e}")

    def _letterbox(self, img, target_size):
        """Resize with padding to preserve aspect ratio."""
        h, w = img.shape[:2]
        scale = min(target_size / w, target_size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x = (target_size - nw) // 2
        pad_y = (target_size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def infer(self, frame):
        """
        Run YOLO on a frame.

        Args:
            frame: BGR numpy array (any size)

        Returns:
            List of dicts: {bbox, class_id, class_name, confidence}
            bbox is (x, y, w, h) in original frame coords.
        """
        if self.net is None:
            return []

        try:
            frame = np.ascontiguousarray(frame)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            letterboxed, scale, pad_x, pad_y = self._letterbox(img_rgb, self.input_size)
            letterboxed = np.ascontiguousarray(letterboxed)

            mat_in = ncnn.Mat.from_pixels(
                letterboxed.tobytes(),
                ncnn.Mat.PixelType.PIXEL_RGB,
                self.input_size, self.input_size,
            )
            mat_in.substract_mean_normalize(
                [0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0],
            )

            ex = self.net.create_extractor()
            ex.input("in0", mat_in)
            _, mat_out = ex.extract("out0")
            out = np.array(mat_out)

            if out.ndim != 2:
                return []

            num_ch, num_anchors = out.shape
            boxes, scores, class_ids = [], [], []

            for i in range(num_anchors):
                cls_scores = out[4:, i]
                class_id = int(np.argmax(cls_scores))
                conf = float(cls_scores[class_id])
                if conf < self.conf_threshold:
                    continue

                cx, cy, bw, bh = out[0, i], out[1, i], out[2, i], out[3, i]
                x1 = (cx - bw / 2 - pad_x) / scale
                y1 = (cy - bh / 2 - pad_y) / scale
                x2 = (cx + bw / 2 - pad_x) / scale
                y2 = (cy + bh / 2 - pad_y) / scale
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(conf)
                class_ids.append(class_id)

            if not boxes:
                return []

            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
            detections = []
            for idx in indices:
                i = int(idx)
                x, y, w, h = [int(v) for v in boxes[i]]
                cid = class_ids[i]
                name = CLASSES[cid] if cid < len(CLASSES) else f"cls{cid}"
                detections.append({
                    "bbox": (x, y, w, h),
                    "class_id": cid,
                    "class_name": name,
                    "confidence": round(scores[i], 3),
                })
            return detections

        except Exception as e:
            print(f"[YOLO] infer error: {e}")
            return []
