"""Pipeline configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Input sources
    sources: List = field(default_factory=lambda: [0, 1])

    # Preprocessing
    proc_width: int = 640
    proc_height: int = 480

    # Motion detection
    motion_history: int = 500
    motion_threshold: float = 16.0
    min_contour_area: float = 500.0

    # YOLO — NCNN
    yolo_ncnn_param: str = "models/yolov26n.param"
    yolo_ncnn_bin: str = "models/yolov26n.bin"
    yolo_conf: float = 0.25
    yolo_nms: float = 0.45
    yolo_input_size: int = 640

    # Autoencoder — ONNX FP16
    ae_model_path: str = "models/autoencoder_fp16.onnx"
    ae_threshold: float = 0.05
    ae_enabled: bool = False
    ae_input_size: int = 128         # 64 for RPi speed, 128 for accuracy
    ae_smooth_window: int = 5        # rolling window size
    ae_smooth_min_hits: int = 2      # require 2/5 frames anomalous
    ae_recheck_interval: int = 15    # re-check tracked regions every N frames

    # Performance
    skip_frames: int = 0
    realtime: bool = False
    profile: bool = False

    # ROI — bottom half of frame (perspective-stable road area)
    roi_top: float = 0.5            # skip top 50% — sky, buildings, horizon
    roi_bottom: float = 1.0

    # Tracker
    tracker_iou: float = 0.25
    tracker_max_lost: int = 10

    # Output
    show_preview: bool = True
    save_video: bool = True
    output_dir: str = "output"

    @staticmethod
    def rpi_preset():
        """Config tuned for RPi 4B (Cortex-A72, 4GB RAM)."""
        cfg = Config()
        cfg.proc_width = 320
        cfg.proc_height = 240
        cfg.motion_history = 200
        cfg.min_contour_area = 200.0
        cfg.yolo_conf = 0.25
        cfg.yolo_input_size = 640
        cfg.ae_enabled = True
        cfg.ae_threshold = 0.06
        cfg.ae_input_size = 64          # 64x64 → <10ms vs 40ms at 128x128
        cfg.ae_smooth_window = 5
        cfg.ae_smooth_min_hits = 2
        cfg.ae_recheck_interval = 15
        cfg.skip_frames = 2
        cfg.roi_top = 0.5              # bottom half — stable road perspective
        cfg.show_preview = False
        cfg.save_video = False
        cfg.tracker_iou = 0.25
        cfg.tracker_max_lost = 15
        return cfg
