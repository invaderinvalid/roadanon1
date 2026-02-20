"""Main pipeline: ties all modules together.

Optimized flow:
  1. Read frame → crop to ROI (skip sky/horizon)
  2. Motion mask on ROI
  3. Black-out motion → AE on masked ROI
  4. Error map → extract crops from ORIGINAL ROI
  5. Feed crops to tracker (IoU matching)
  6. YOLO runs ONLY on NEW tracks (biggest FPS win)
  7. Active tracks carry forward labels — no re-classification
  8. Log + save image once per track lifetime
"""

import cv2
import json
import os
import argparse
import time
import numpy as np
from datetime import datetime
from config import Config
from preprocessing import MultiCamera
from motion_detect import MotionDetector
from autoencoder import AnomalyDetector
from cropping import Cropper, Validator
from tracker import SimpleTracker


# ═══════════════════════════════════════════════════════════════
# Classifier loader
# ═══════════════════════════════════════════════════════════════

def load_classifier(cfg: Config):
    """Pick classifier backend based on config."""
    if cfg.classifier_backend == "ncnn":
        from classifier_ncnn import YOLOClassifierNCNN
        return YOLOClassifierNCNN(cfg)
    elif cfg.classifier_backend == "torch":
        from classifier_torch import YOLOClassifierTorch
        return YOLOClassifierTorch(cfg)
    else:  # tflite
        from classifier import YOLOClassifier
        return YOLOClassifier(cfg)


# ═══════════════════════════════════════════════════════════════
# Drawing
# ═══════════════════════════════════════════════════════════════

YOLO_COLORS = [
    (0, 0, 255),    # road_damage  — red
    (0, 165, 255),  # speed_bump   — orange
    (0, 255, 255),  # unsurfaced   — yellow
]

CLASS_COLORS = {
    "road_damage": (0, 0, 255),
    "speed_bump": (0, 165, 255),
    "unsurfaced_road": (0, 255, 255),
}


def draw_results(frame, roi_y, active_tracks, mean_error, fps, motion_regions=None):
    display = frame.copy()
    h, w = display.shape[:2]

    # ROI boundary (dotted cyan line)
    if roi_y > 0:
        for x0 in range(0, w, 12):
            cv2.line(display, (x0, roi_y), (min(x0 + 6, w), roi_y), (255, 255, 0), 1)

    # Motion regions (dim green, within ROI)
    if motion_regions:
        for (mx, my, mw, mh), _ in motion_regions:
            cv2.rectangle(display, (mx, roi_y + my), (mx + mw, roi_y + my + mh),
                          (0, 180, 0), 1)

    # Tracked anomalies
    for t in active_tracks:
        tx, ty, tw, th = t.bbox
        # Offset bbox back to full-frame coords
        fy = roi_y + ty
        is_known = t.label is not None
        color = CLASS_COLORS.get(t.label, (255, 255, 0))  # cyan for unknown
        thickness = 2 if is_known else 1

        cv2.rectangle(display, (tx, fy), (tx + tw, fy + th), color, thickness)

        # Label
        if is_known:
            lbl = f"T{t.id} {t.label} {t.confidence:.2f}"
        else:
            lbl = f"T{t.id} unknown AE:{t.score:.3f}"
        cv2.putText(display, lbl, (tx, max(fy - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # YOLO sub-boxes (only for known)
        for det in t.yolo_dets:
            bx, by, bw, bh = det["bbox"]
            bx2, by2 = tx + bx, fy + by
            dc = CLASS_COLORS.get(det["class_name"], (255, 255, 0))
            cv2.rectangle(display, (bx2, by2), (bx2 + bw, by2 + bh), dc, 2)

    # Status bar
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, f"AE: {mean_error:.4f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2)
    cv2.putText(display, f"Tracks: {len(active_tracks)}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return display


# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════

class JSONLLogger:
    """Writes one JSON line per NEW track (first detection only)."""

    def __init__(self, path):
        self._file = open(path, "w")

    def log_track(self, track, cam_id):
        """Log when a track is first confirmed. Returns event dict."""
        top_cls = track.label or "unknown_anomaly"
        top_conf = track.confidence

        event = {
            "timestamp": datetime.now().isoformat(),
            "track_id": track.id,
            "first_frame": track.first_frame,
            "camera": cam_id,
            "kind": "known" if track.label else "unknown",
            "ae_error": round(track.score, 4),
            "bbox": {"x": track.bbox[0], "y": track.bbox[1],
                     "w": track.bbox[2], "h": track.bbox[3]},
            "class": top_cls,
            "yolo_confidence": top_conf,
            "yolo_detections": [
                {"class": d["class_name"], "confidence": d["confidence"],
                 "bbox": {"x": d["bbox"][0], "y": d["bbox"][1],
                          "w": d["bbox"][2], "h": d["bbox"][3]}}
                for d in track.yolo_dets
            ],
        }
        self._file.write(json.dumps(event) + "\n")
        return event

    def close(self):
        self._file.close()


class VideoWriterWrapper:
    """Per-camera annotated video output."""

    def __init__(self, output_dir, width, height, fps=15.0):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self._writers = {}

    def write(self, cam_id, frame):
        if cam_id not in self._writers:
            path = os.path.join(self.output_dir, f"cam{cam_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writers[cam_id] = cv2.VideoWriter(path, fourcc, self.fps,
                                                     (self.width, self.height))
            print(f"[Video] cam{cam_id} → {path}")
        out = frame
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            out = cv2.resize(frame, (self.width, self.height))
        self._writers[cam_id].write(out)

    def release(self):
        for w in self._writers.values():
            w.release()


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════

def run(cfg: Config):
    print("╔══════════════════════════════════════╗")
    print("║   Road Anomaly Detection Pipeline    ║")
    print("║   (ROI + Tracker optimized)          ║")
    print("╚══════════════════════════════════════╝\n")

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "detections.jsonl")

    camera = MultiCamera(cfg)
    src_fps = camera.src_fps or 30.0

    # ROI: vertical slice of frame (skip sky/horizon)
    roi_y_start = int(cfg.proc_height * cfg.roi_top)     # e.g. 0.4 * 480 = 192
    roi_y_end = int(cfg.proc_height * cfg.roi_bottom)     # e.g. 1.0 * 480 = 480
    roi_h = roi_y_end - roi_y_start

    print(f"[ROI] y={roi_y_start}..{roi_y_end} ({roi_h}px of {cfg.proc_height}px) "
          f"— skipping top {cfg.roi_top*100:.0f}%")

    # Per-camera modules
    motion_dets = [MotionDetector(cfg) for _ in cfg.sources]
    trackers = [SimpleTracker(iou_thresh=cfg.tracker_iou,
                              max_lost=cfg.tracker_max_lost)
                for _ in cfg.sources]

    ae = AnomalyDetector(cfg)
    cropper = Cropper(cfg)
    validator = Validator(cfg)
    yolo = load_classifier(cfg)
    logger = JSONLLogger(log_path)
    vid_writer = VideoWriterWrapper(cfg.output_dir, cfg.proc_width, cfg.proc_height) if cfg.save_video else None

    print(f"[Pipeline] Log → {log_path}")
    print(f"[Pipeline] Tracker IoU={cfg.tracker_iou} max_lost={cfg.tracker_max_lost}")
    print(f"[Pipeline] Video: {'ON' if cfg.save_video else 'OFF (--no-video)'}")
    print("[Pipeline] Running... Press 'q' to quit.\n")

    frame_count = 0
    skip_n = cfg.skip_frames
    known_total = 0
    unknown_total = 0
    yolo_calls = 0
    t_start = time.time()
    fps = 0.0

    if skip_n > 0:
        print(f"[Pipeline] Frame skip: processing every {skip_n} frame(s)")

    try:
        while True:
            frames = camera.read_all()
            if all(f is None for f in frames):
                print("[Pipeline] All sources exhausted.")
                break

            frame_count += 1

            # Frame skipping — update BG model but skip heavy ops
            if skip_n > 0 and (frame_count % skip_n != 0):
                for idx, frame in enumerate(frames):
                    if frame is not None:
                        roi = frame[roi_y_start:roi_y_end]
                        motion_dets[idx].detect(roi)
                continue

            for idx, frame in enumerate(frames):
                if frame is None:
                    continue

                t_frame = time.time()
                mean_error = 0.0
                motion_regions = []

                # ── Stage 1: Crop to ROI ──
                roi = frame[roi_y_start:roi_y_end]

                # ── Stage 2: Motion mask on ROI ──
                mask, motion_regions = motion_dets[idx].detect(roi)

                # ── Build detection candidates for tracker ──
                det_candidates = []

                if motion_regions:
                    # ── Stage 3: Mask moving objects before AE (in-place, no copy) ──
                    mask_bool = mask > 0
                    masked_roi = roi.copy()  # need copy since roi is a view
                    masked_roi[mask_bool] = 0

                    # ── Stage 4: AE on masked ROI ──
                    error_map, mean_error = ae.infer(masked_roi)
                    error_map[mask_bool] = 0.0
                    unmasked = ~mask_bool
                    if unmasked.any():
                        mean_error = float(error_map[unmasked].mean())
                    else:
                        mean_error = 0.0

                    if mean_error > cfg.ae_threshold:
                        # ── Stage 5: Crop from ORIGINAL ROI ──
                        crops = cropper.extract(roi, error_map, cfg.ae_threshold)

                        crop_count = 0
                        for crop in crops:
                            if crop_count >= cfg.max_crops:
                                break
                            if validator.is_valid(crop["image"]):
                                det_candidates.append({
                                    "bbox": crop["bbox"],      # in ROI coords
                                    "score": crop["score"],
                                    "image": crop["image"],
                                })
                                crop_count += 1

                # ── Stage 6: Tracker update ──
                tracker = trackers[idx]
                new_tracks, active_tracks, gone_tracks = tracker.update(
                    det_candidates, frame_count)

                # ── Stage 7: YOLO only on NEW tracks ──
                for t in new_tracks:
                    if t.image is not None and validator.is_valid(t.image):
                        dets = yolo.infer(t.image)
                        yolo_calls += 1
                        t.yolo_dets = dets
                        if dets:
                            t.label = dets[0]["class_name"]
                            t.confidence = dets[0]["confidence"]

                    # ── Stage 8: Log (once per track, no image saving) ──
                    t.logged = True
                    event = logger.log_track(t, idx)
                    if t.label:
                        known_total += 1
                    else:
                        unknown_total += 1

                # ── Draw + write video (conditional) ──
                elapsed = time.time() - t_frame
                fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))

                if cfg.save_video or cfg.show_preview:
                    display = draw_results(frame, roi_y_start, active_tracks,
                                           mean_error, fps, motion_regions)
                    if vid_writer:
                        vid_writer.write(idx, display)
                    if cfg.show_preview:
                        cv2.imshow(f"Camera {idx}", display)

            if frame_count % 30 == 0:
                total_tracks = sum(len(tr.tracks) for tr in trackers)
                print(f"  Frame {frame_count} | FPS: {fps:.1f} | "
                      f"Tracks: {total_tracks} | YOLO calls: {yolo_calls}")

            if cfg.show_preview:
                if cfg.realtime:
                    proc_ms = int((time.time() - t_frame) * 1000)
                    wait_ms = max(1, int(1000 / src_fps) - proc_ms)
                else:
                    wait_ms = 1
                key = cv2.waitKey(wait_ms)
                if key == ord("q") or key == 27:
                    print("\n[Pipeline] Quit requested")
                    break

    finally:
        camera.release()
        if vid_writer:
            vid_writer.release()
        logger.close()
        cv2.destroyAllWindows()

    elapsed_total = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Pipeline Complete!")
    print(f"{'='*55}")
    print(f"  Total frames:      {frame_count}")
    print(f"  Known anomalies:   {known_total}")
    print(f"  Unknown anomalies: {unknown_total}")
    print(f"  Total YOLO calls:  {yolo_calls} (vs {frame_count} frames)")
    print(f"  Avg FPS:           {frame_count / max(elapsed_total, 0.001):.1f}")
    print(f"  Time:              {elapsed_total:.1f}s")
    print(f"  ROI:               {cfg.roi_top*100:.0f}%-{cfg.roi_bottom*100:.0f}% "
          f"({roi_h}px)")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="Road Anomaly Detection Pipeline")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Video sources (camera indices or file paths)")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Alias for --no-preview")
    parser.add_argument("--ae-threshold", type=float, default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--backend", choices=["ncnn", "torch", "tflite"], default=None,
                        help="YOLO classifier backend")
    parser.add_argument("--ae-backend", choices=["torch", "onnx"], default=None,
                        help="Autoencoder backend (onnx for RPi 4B)")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--skip-frames", type=int, default=None,
                        help="Process every Nth frame (0=all)")
    parser.add_argument("--max-crops", type=int, default=None,
                        help="Max anomaly crops sent to YOLO per frame")
    parser.add_argument("--realtime", action="store_true",
                        help="Throttle display to source video FPS")
    parser.add_argument("--proc-width", type=int, default=None)
    parser.add_argument("--proc-height", type=int, default=None)
    parser.add_argument("--roi-top", type=float, default=None,
                        help="Skip top N%% of frame (0.0-1.0, default 0.4)")
    parser.add_argument("--no-tracker", action="store_true",
                        help="Disable IoU tracker (run YOLO every frame)")
    parser.add_argument("--tracker-iou", type=float, default=None,
                        help="Tracker IoU threshold (default 0.25)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video encoding (saves CPU on RPi)")
    parser.add_argument("--rpi", action="store_true",
                        help="RPi 4B preset: 320x240, onnx, ncnn, skip=2, no video")
    args = parser.parse_args()

    cfg = Config.rpi_preset() if args.rpi else Config()
    if args.sources:
        cfg.sources = [int(s) if s.isdigit() else s for s in args.sources]
    if args.no_preview or args.headless:
        cfg.show_preview = False
    if args.ae_threshold is not None:
        cfg.ae_threshold = args.ae_threshold
    if args.yolo_conf is not None:
        cfg.yolo_conf = args.yolo_conf
    if args.backend:
        cfg.classifier_backend = args.backend
    if args.ae_backend:
        cfg.ae_backend = args.ae_backend
    if args.output:
        cfg.output_dir = args.output
    if args.skip_frames is not None:
        cfg.skip_frames = args.skip_frames
    if args.max_crops is not None:
        cfg.max_crops = args.max_crops
    if args.realtime:
        cfg.realtime = True
    if args.proc_width is not None:
        cfg.proc_width = args.proc_width
    if args.proc_height is not None:
        cfg.proc_height = args.proc_height
    if args.roi_top is not None:
        cfg.roi_top = args.roi_top
    if args.no_tracker:
        cfg.tracker_enabled = False
    if args.tracker_iou is not None:
        cfg.tracker_iou = args.tracker_iou
    if args.no_video:
        cfg.save_video = False

    run(cfg)


if __name__ == "__main__":
    main()
