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

    # YOLO — NCNN
    yolo_ncnn_param: str = "models/yolov26n.param"
    yolo_ncnn_bin: str = "models/yolov26n.bin"
    yolo_conf: float = 0.25
    yolo_nms: float = 0.45
    yolo_input_size: int = 640

    # Performance
    skip_frames: int = 0        # 0 = process every frame, N = process every Nth
    realtime: bool = False      # throttle display to source FPS
    profile: bool = False       # print per-stage timing

    # Motion gate
    motion_min_pct: float = 0.5     # skip if motion < 0.5% of ROI (noise)
    motion_max_pct: float = 60.0    # skip if motion > 60% of ROI (shake)

    # ROI — restrict processing to road area
    roi_top: float = 0.4       # skip top 40% (sky/horizon)
    roi_bottom: float = 1.0

    # Tracker — dedup same anomaly across frames
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
        cfg.motion_history = 200       # stable BG model without being slow
        cfg.min_contour_area = 200.0   # catch smaller anomalies at 320px
        cfg.yolo_conf = 0.25           # don't filter too aggressively
        cfg.yolo_input_size = 640
        cfg.skip_frames = 2            # process every 2nd frame (not 4th)
        cfg.roi_top = 0.35
        cfg.show_preview = False
        cfg.save_video = False
        cfg.tracker_iou = 0.25         # stricter matching = fewer merged tracks
        cfg.tracker_max_lost = 15
        return cfg
