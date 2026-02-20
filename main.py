"""Road Anomaly Detection — GUI (OpenCV-based).

Two modes:
  [1] Pi Camera / USB webcam (live)
  [2] Video file playback

Shows live video with detections, AE/YOLO status, and anomaly alerts.
Uses pipeline modules directly — no pipeline.py modification.

Controls:
  q / ESC  — quit
  SPACE    — pause/resume
  r        — reset tracker
"""

import cv2
import numpy as np
import time
import os
import sys
import glob
from datetime import datetime
from collections import deque

from config import Config
from motion_detect import MotionDetector
from tracker import SimpleTracker, _iou


# ═══════════════════════════════════════════════════════════════
# HUD Drawing Helpers
# ═══════════════════════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1, alpha=0.7):
    """Draw a semi-transparent rounded rectangle."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_bar(img, x, y, w, h, value, max_val, color, bg=(40, 40, 40), thresh=None):
    """Draw a horizontal bar with optional threshold marker."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fill = int(min(value / max(max_val, 0.001), 1.0) * w)
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    if thresh is not None:
        tx = x + int(min(thresh / max(max_val, 0.001), 1.0) * w)
        cv2.line(img, (tx, y - 2), (tx, y + h + 2), (255, 255, 255), 1)


def put_text(img, text, pos, scale=0.4, color=(220, 220, 220), thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


CLASS_COLORS = {
    "road_damage": (60, 60, 255),
    "speed_bump": (0, 165, 255),
    "unsurfaced_road": (220, 220, 0),
}


# ═══════════════════════════════════════════════════════════════
# Main GUI
# ═══════════════════════════════════════════════════════════════

def select_mode():
    """Terminal-based mode selection."""
    print("\n╔══════════════════════════════════════╗")
    print("║   Road Anomaly Detection — GUI       ║")
    print("╚══════════════════════════════════════╝\n")
    print("  Select mode:")
    print("    [1] Pi Camera / USB Webcam")
    print("    [2] Video file")
    print("    [3] All videos in test_video/")
    print()

    choice = input("  Mode (1/2/3): ").strip()

    if choice == "1":
        cam_id = input("  Camera ID [0]: ").strip() or "0"
        return int(cam_id), "camera"
    elif choice == "2":
        path = input("  Video path: ").strip()
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found")
            sys.exit(1)
        return path, "video"
    elif choice == "3":
        videos = sorted(glob.glob("test_video/*.mp4"))
        if not videos:
            print("  ERROR: No .mp4 files in test_video/")
            sys.exit(1)
        print(f"  Found {len(videos)} videos")
        return videos, "batch"
    else:
        print("  Invalid choice")
        sys.exit(1)


def run_gui(source, mode="video"):
    """Run detection with OpenCV GUI overlay."""

    cfg = Config()
    cfg.show_preview = False
    cfg.save_video = False
    cfg.motion_history = 200
    cfg.min_contour_area = 300
    cfg.yolo_conf = 0.25
    cfg.roi_top = 0.5
    cfg.ae_enabled = True
    cfg.ae_input_size = 64
    cfg.ae_threshold = 0.06
    cfg.ae_smooth_window = 5
    cfg.ae_smooth_min_hits = 2
    cfg.ae_recheck_interval = 15
    cfg.skip_frames = 0
    cfg.tracker_iou = 0.25
    cfg.tracker_max_lost = 15

    if mode == "camera":
        cfg.proc_width = 320
        cfg.proc_height = 240
    else:
        cfg.proc_width = 640
        cfg.proc_height = 480

    # Open source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {source}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode != "camera" else 0
    src_name = os.path.basename(str(source)) if isinstance(source, str) else f"Camera {source}"

    # Init modules
    from classifier_ncnn import YOLOClassifier
    cfg.sources = [source]
    yolo = YOLOClassifier(cfg)
    yolo.start_async()

    ae = None
    if cfg.ae_enabled:
        try:
            from autoencoder_tflite import AutoencoderDetector
            ae = AutoencoderDetector(cfg)
            if ae.session is None:
                ae = None
        except Exception:
            ae = None

    motion_det = MotionDetector(cfg)
    tracker = SimpleTracker(iou_thresh=cfg.tracker_iou, max_lost=cfg.tracker_max_lost)

    roi_y_start = int(cfg.proc_height * cfg.roi_top)
    roi_y_end = int(cfg.proc_height * cfg.roi_bottom)
    roi_h = roi_y_end - roi_y_start
    roi_area = max(roi_h * cfg.proc_width, 1)

    # State
    frame_count = 0
    yolo_calls = 0
    fps = 0.0
    load_avg_ms = 0.0
    last_ae_error = None
    yolo_cooldown = 0
    paused = False

    # Alert log (last 6 alerts)
    alerts = deque(maxlen=6)

    # Display window
    win_name = "Road Anomaly Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 600)

    print(f"\n  Playing: {src_name}")
    print(f"  Controls: [q]uit  [SPACE]pause  [r]eset tracker\n")

    while True:
        if not paused:
            ret, raw_frame = cap.read()
            if not ret:
                break

            frame_count += 1
            t0 = time.time()

            frame = cv2.resize(raw_frame, (cfg.proc_width, cfg.proc_height))
            roi = frame[roi_y_start:roi_y_end]

            # ── Motion ──
            mask, motion_regions = motion_det.detect(roi)
            motion_pct = 0.0
            if motion_regions:
                total_m = sum(a for _, a in motion_regions)
                motion_pct = (total_m / roi_area) * 100.0

            if motion_pct < 0.1:
                mw = 3.0
            elif motion_pct < 2.0:
                mw = 1.5
            elif motion_pct > 50.0:
                mw = 1.8
            elif motion_pct > 30.0:
                mw = 1.3
            else:
                mw = 1.0

            # ── Tracker check ──
            all_tracked = True
            if motion_regions:
                for (mx, my, mw2, mh), _ in motion_regions:
                    if not any(_iou((mx, my, mw2, mh), tb) > 0.3
                               for tb in tracker.active_bboxes):
                        all_tracked = False
                        break

            # ── AE ──
            ae_says_anomaly = False
            run_ae = False
            if ae and motion_regions:
                run_ae = not all_tracked or (frame_count % cfg.ae_recheck_interval == 0)
                if run_ae:
                    eff_thresh = cfg.ae_threshold * mw
                    ae_says_anomaly, ae_err, _ = ae.is_anomaly_smoothed(roi, eff_thresh)
                    last_ae_error = ae_err

            # ── YOLO fetch ──
            yolo_result = yolo.fetch()
            det_candidates = []
            if yolo_result:
                for det in yolo_result:
                    bx, by, bw, bh = det["bbox"]
                    det_candidates.append({
                        "bbox": (bx, by, bw, bh),
                        "score": det["confidence"],
                        "image": None,
                        "label": det["class_name"],
                        "confidence": det["confidence"],
                    })

            # ── YOLO submit ──
            submit = False
            if not all_tracked:
                submit = ae_says_anomaly if ae else bool(motion_regions)
            elif ae_says_anomaly and frame_count % cfg.ae_recheck_interval == 0:
                submit = True
            if yolo_cooldown > 0:
                yolo_cooldown -= 1
                submit = False
            if submit:
                yolo.submit(roi)
                yolo_calls += 1
                yolo_cooldown = 5

            # ── Tracker ──
            new_tracks, active_tracks, gone_tracks = tracker.update(
                det_candidates, frame_count)
            for t in new_tracks:
                for dc in det_candidates:
                    if dc["bbox"] == t.bbox:
                        t.label = dc["label"]
                        t.confidence = dc["confidence"]
                        break

            elapsed = time.time() - t0
            elapsed_ms = elapsed * 1000
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))
            load_avg_ms = 0.9 * load_avg_ms + 0.1 * elapsed_ms

            # ── Generate alerts ──
            for t in new_tracks:
                if t.label:
                    ts = datetime.now().strftime("%H:%M:%S")
                    alerts.append((ts, t.label, t.confidence, t.id))

        # ═══════════════════════════════════════════════════════
        # DRAW
        # ═══════════════════════════════════════════════════════
        display = frame.copy()
        h, w = display.shape[:2]

        # Upscale for better readability
        scale = 2 if w < 400 else 1
        if scale > 1:
            display = cv2.resize(display, (w * scale, h * scale),
                                 interpolation=cv2.INTER_NEAREST)
        dh, dw = display.shape[:2]
        ry = roi_y_start * scale

        # ROI line
        for x0 in range(0, dw, 16):
            cv2.line(display, (x0, ry), (min(x0 + 8, dw), ry), (0, 255, 255), 1)

        # Motion boxes
        if motion_regions:
            for (mx, my, mw2, mh), _ in motion_regions:
                cv2.rectangle(display,
                              (mx * scale, ry + my * scale),
                              ((mx + mw2) * scale, ry + (my + mh) * scale),
                              (0, 180, 0), 1)

        # Track boxes
        for t in active_tracks:
            tx, ty, tw, th = t.bbox
            fy = ry + ty * scale
            color = CLASS_COLORS.get(t.label, (0, 255, 255))
            cv2.rectangle(display,
                          (tx * scale, fy), ((tx + tw) * scale, fy + th * scale),
                          color, 2)
            if t.label:
                lbl = f"{t.label} {t.confidence:.0%}"
            else:
                lbl = f"T{t.id}"
            put_text(display, lbl, (tx * scale, max(fy - 8, 16)), 0.5, color, 1)

        # ── HUD Panel (right side) ──
        panel_w = 220
        panel_x = dw - panel_w - 8
        panel_y = 8
        draw_rounded_rect(display, (panel_x, panel_y),
                          (panel_x + panel_w, panel_y + 200),
                          (15, 15, 30), alpha=0.75)

        px = panel_x + 10
        py = panel_y + 18
        lh = 20  # line height

        # Title
        put_text(display, src_name[:25], (px, py), 0.4, (0, 212, 255), 1)
        py += lh

        # FPS + latency
        lat_color = (0, 220, 0) if load_avg_ms < 33 else (0, 200, 255) if load_avg_ms < 66 else (0, 0, 255)
        put_text(display, f"FPS: {fps:.0f}", (px, py), 0.45, (255, 255, 255), 1)
        put_text(display, f"{load_avg_ms:.0f}ms", (px + 80, py), 0.4, lat_color, 1)
        py += lh

        # Frame progress
        if total_frames > 0:
            pct = frame_count / total_frames
            put_text(display, f"Frame: {frame_count}/{total_frames}", (px, py), 0.35, (160, 160, 160))
            draw_bar(display, px, py + 4, panel_w - 20, 4, pct, 1.0, (0, 180, 255))
        else:
            put_text(display, f"Frame: {frame_count}", (px, py), 0.35, (160, 160, 160))
        py += lh

        # AE error bar
        if last_ae_error is not None:
            ae_col = (0, 0, 255) if ae_says_anomaly else (0, 220, 0)
            thresh_val = cfg.ae_threshold * mw if ae else 0
            max_ae = max(thresh_val * 3, 0.15)
            draw_bar(display, px, py - 4, panel_w - 20, 8,
                     last_ae_error, max_ae, ae_col, thresh=thresh_val)
            ae_star = " *" if run_ae else ""
            put_text(display, f"AE: {last_ae_error:.4f}{ae_star}", (px, py + 12), 0.35, ae_col)
        else:
            put_text(display, "AE: off", (px, py + 5), 0.35, (100, 100, 100))
        py += lh + 8

        # YOLO status
        if yolo.is_busy:
            yolo_str, yolo_col = f"YOLO: BUSY ({yolo_calls})", (0, 200, 255)
        elif yolo_cooldown > 0:
            yolo_str, yolo_col = f"YOLO: CD:{yolo_cooldown} ({yolo_calls})", (255, 200, 0)
        else:
            yolo_str, yolo_col = f"YOLO: READY ({yolo_calls})", (0, 220, 0)
        put_text(display, yolo_str, (px, py), 0.38, yolo_col)
        py += lh

        # Motion + tracks
        put_text(display, f"Motion: {motion_pct:.1f}%  Tracks: {len(active_tracks)}",
                 (px, py), 0.35, (160, 160, 160))
        py += lh

        # Anomaly indicator
        if ae_says_anomaly:
            put_text(display, "!! ANOMALY !!", (px, py), 0.5, (0, 0, 255), 2)
        py += lh

        # ── Alert panel (bottom) ──
        if alerts:
            alert_h = min(len(alerts), 4) * 22 + 10
            alert_y = dh - alert_h - 8
            draw_rounded_rect(display, (8, alert_y), (dw - 8, dh - 8),
                              (10, 10, 40), alpha=0.75)
            ay = alert_y + 18
            for ts, label, conf, tid in list(alerts)[-4:]:
                color = CLASS_COLORS.get(label, (0, 255, 255))
                put_text(display, f"[{ts}]", (16, ay), 0.32, (100, 100, 100))
                put_text(display, f" {label} {conf:.0%} #T{tid}",
                         (90, ay), 0.38, color, 1)
                ay += 22

        # Pause indicator
        if paused:
            put_text(display, "|| PAUSED", (dw // 2 - 50, dh // 2), 0.8,
                     (255, 255, 255), 2)

        cv2.imshow(win_name, display)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            tracker = SimpleTracker(iou_thresh=cfg.tracker_iou,
                                     max_lost=cfg.tracker_max_lost)
            alerts.clear()

    # Cleanup
    yolo.stop_async()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Done: {frame_count} frames, {yolo_calls} YOLO calls, "
          f"avg FPS: {fps:.1f}\n")


def main():
    source, mode = select_mode()

    if mode == "batch":
        for video_path in source:
            print(f"\n{'─'*50}")
            run_gui(video_path, "video")
    else:
        run_gui(source, mode)


if __name__ == "__main__":
    main()
