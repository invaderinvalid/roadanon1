"""Pipeline configuration."""

from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Input sources (camera indices or video file paths)
    sources: List = field(default_factory=lambda: [0, 1])

    # Preprocessing
    proc_width: int = 640
    proc_height: int = 480

    # Motion detection
    motion_history: int = 500
    motion_threshold: float = 16.0
    min_contour_area: float = 500.0

    # Autoencoder
    autoencoder_path: str = "models/model.pth"
    autoencoder_onnx_path: str = "models/model.onnx"
    autoencoder_int8_path: str = "models/model_int8.onnx"  # INT8 quantized
    ae_backend: str = "torch"  # "torch" (Mac/dev) or "onnx" (RPi deploy)
    ae_threshold: float = 0.05
    ae_use_int8: bool = False    # use quantized model (2-4× faster on ARM)

    # Cropping / Validation
    crop_padding: int = 10
    crop_min_size: int = 32
    black_max_ratio: float = 0.85
    black_pixel_thresh: int = 15

    # YOLO — NCNN (Mac/dev)
    yolo_ncnn_param: str = "models/yolov26n.param"
    yolo_ncnn_bin: str = "models/yolov26n.bin"
    yolo_conf: float = 0.005
    yolo_nms: float = 0.45
    yolo_input_size: int = 640

    # YOLO — TFLite (RPi deploy)
    yolo_tflite_path: str = "models/yolov26.tflite"

    # YOLO — PyTorch (fallback dev)
    yolo_pt_path: str = "models/best.pt"

    # Classifier backend: "ncnn", "torch", "tflite"
    classifier_backend: str = "ncnn"

    # Performance
    skip_frames: int = 0       # 0 = process every frame, N = process every Nth
    max_crops: int = 5         # max anomaly crops sent to YOLO per frame
    realtime: bool = False     # True = throttle display to source FPS
    profile: bool = False      # print per-stage timing

    # Motion gate — skip AE when motion is too small or too large
    motion_min_pct: float = 0.5    # skip AE if motion < 0.5% of ROI (noise)
    motion_max_pct: float = 60.0   # skip AE if motion > 60% of ROI (camera shake)

    # ROI — restrict processing to road area (fraction of frame height)
    roi_top: float = 0.4       # 0.0=full frame top, 0.4=skip top 40% (sky/horizon)
    roi_bottom: float = 1.0    # 1.0=full frame bottom

    # Tracker — dedup same anomaly across frames
    tracker_iou: float = 0.25  # IoU threshold for matching
    tracker_max_lost: int = 10 # frames before dropping a track
    tracker_enabled: bool = True

    # Output
    show_preview: bool = True
    save_video: bool = True    # False = skip mp4 encoding (big RPi speedup)
    output_dir: str = "output"
    log_format: str = "jsonl"  # "jsonl" or "csv"

    @staticmethod
    def rpi_preset():
        """Conservative config tuned for RPi 4B (Cortex-A72, 4GB RAM)."""
        cfg = Config()
        cfg.proc_width = 320
        cfg.proc_height = 240
        cfg.ae_backend = "onnx"         # no torch → no SIGILL
        cfg.ae_use_int8 = True           # INT8 quantized (2-4× faster on ARM)
        cfg.ae_threshold = 0.08          # higher = fewer false positives
        cfg.classifier_backend = "ncnn"  # or "tflite"
        cfg.yolo_conf = 0.35             # reject low-confidence garbage
        cfg.yolo_input_size = 640        # must match NCNN model export size
        cfg.skip_frames = 4              # process every 4th frame
        cfg.max_crops = 3                # cap YOLO calls per frame
        cfg.crop_min_size = 48           # skip tiny crops
        cfg.roi_top = 0.35               # skip top 35%
        cfg.show_preview = False
        cfg.save_video = False           # skip mp4 encoding
        cfg.tracker_iou = 0.2
        cfg.tracker_max_lost = 15
        return cfg
