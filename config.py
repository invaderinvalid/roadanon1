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
    ae_backend: str = "torch"  # "torch" (Mac/dev) or "onnx" (RPi deploy)
    ae_threshold: float = 0.05

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

    # Output
    show_preview: bool = True
    output_dir: str = "output"
    log_format: str = "jsonl"  # "jsonl" or "csv"
