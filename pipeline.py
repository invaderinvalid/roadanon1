"""Road Anomaly Detection Pipeline — Real-time on RPi 4B.

Fixes applied:
  1. Tracker only bypasses YOLO, not AE. AE periodically re-checks.
  2. YOLO queue max=1, latest frame always wins (no overflow).
  3. Backpressure: degrade AE frequency when CPU overloaded.
  4. AE input 64x64 → <10ms (under 30ms budget).
  5. ROI = bottom half (perspective-stable road).
  6. Temporal smoothing: require 2/5 frames anomalous before YOLO.

Thread layout (RPi 4B — 4 cores):
  Core 1: Camera capture thread
  Core 2: Main thread (motion + AE + tracker + draw)
  Core 3-4: YOLO NCNN thread (async, 2 internal threads)
"""

import cv2
import json
import os
import gc
import argparse
import time
import numpy as np
from datetime import datetime
from collections import defaultdict
from config import Config
from preprocessing import MultiCamera
from motion_detect import MotionDetector
from tracker import SimpleTracker, _iou


# ═══════════════════════════════════════════════════════════════
# Drawing
# ═══════════════════════════════════════════════════════════════

CLASS_COLORS = {
    "road_damage": (0, 0, 255),
    "speed_bump": (0, 165, 255),
    "unsurfaced_road": (0, 255, 255),
}


def draw_results(frame, roi_y, active_tracks, fps, motion_regions=None,
                 hud=None):
    """Draw rich overlay with AE, YOLO, CPU stats.

    hud dict keys:
        ae_error, ae_threshold, ae_active,
        yolo_busy, yolo_cooldown, yolo_calls,
        load_ms, motion_pct, frame_count
    """
    display = frame.copy()
    h, w = display.shape[:2]
    hud = hud or {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    gray = (160, 160, 160)
    green = (0, 220, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    cyan = (255, 200, 0)

    # ── ROI line ──
    if roi_y > 0:
        for x0 in range(0, w, 12):
            cv2.line(display, (x0, roi_y), (min(x0 + 6, w), roi_y), yellow, 1)

    # ── Motion regions ──
    if motion_regions:
        for (mx, my, mw, mh), _ in motion_regions:
            cv2.rectangle(display, (mx, roi_y + my), (mx + mw, roi_y + my + mh),
                          (0, 180, 0), 1)

    # ── Track boxes with label + confidence ──
    for t in active_tracks:
        tx, ty, tw, th = t.bbox
        fy = roi_y + ty
        color = CLASS_COLORS.get(t.label, yellow)
        thickness = 2 if t.label else 1
        cv2.rectangle(display, (tx, fy), (tx + tw, fy + th), color, thickness)

        # Label line 1: class + confidence
        if t.label:
            lbl = f"{t.label} {t.confidence:.0%}"
        else:
            lbl = f"T{t.id} ???"
        cv2.putText(display, lbl, (tx, max(fy - 5, 12)), font, 0.38, color, 1)

        # Label line 2: track ID
        cv2.putText(display, f"#{t.id}", (tx, fy + th + 12), font, 0.3, gray, 1)

    # ── HUD panel (top-left) ──
    y = 14
    line_h = 16

    # Row 1: FPS + frame latency
    load_ms = hud.get('load_ms', 0)
    latency_color = green if load_ms < 33 else yellow if load_ms < 66 else red
    cv2.putText(display, f"FPS {fps:.0f}", (4, y), font, 0.45, white, 1)
    cv2.putText(display, f"{load_ms:.0f}ms", (70, y), font, 0.38, latency_color, 1)
    y += line_h

    # Row 2: AE error + threshold bar
    ae_err = hud.get('ae_error')
    ae_thresh = hud.get('ae_threshold', 0)
    if ae_err is not None:
        # Mini bar: |====|--T--| where = is error, T is threshold
        bar_w = 80
        bar_h = 8
        bx, by = 4, y - 6
        max_val = max(ae_thresh * 3, 0.15)  # scale
        err_px = int(min(ae_err / max_val, 1.0) * bar_w)
        thresh_px = int(min(ae_thresh / max_val, 1.0) * bar_w)

        # Background
        cv2.rectangle(display, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), -1)
        # Error fill
        err_color = red if ae_err > ae_thresh else green
        cv2.rectangle(display, (bx, by), (bx + err_px, by + bar_h), err_color, -1)
        # Threshold marker
        cv2.line(display, (bx + thresh_px, by - 2), (bx + thresh_px, by + bar_h + 2), white, 1)

        ae_label = f"AE {ae_err:.3f}"
        if hud.get('ae_active'):
            ae_label += " *"
        cv2.putText(display, ae_label, (bx + bar_w + 4, y), font, 0.35, err_color, 1)
    else:
        cv2.putText(display, "AE off", (4, y), font, 0.35, gray, 1)
    y += line_h

    # Row 3: YOLO status
    yolo_busy = hud.get('yolo_busy', False)
    yolo_cd = hud.get('yolo_cooldown', 0)
    yolo_n = hud.get('yolo_calls', 0)
    if yolo_busy:
        yolo_str = f"YOLO: BUSY ({yolo_n})"
        yolo_color = yellow
    elif yolo_cd > 0:
        yolo_str = f"YOLO: CD {yolo_cd} ({yolo_n})"
        yolo_color = cyan
    else:
        yolo_str = f"YOLO: READY ({yolo_n})"
        yolo_color = green
    cv2.putText(display, yolo_str, (4, y), font, 0.35, yolo_color, 1)
    y += line_h

    # Row 4: Motion % + track count
    motion_pct = hud.get('motion_pct', 0)
    n_tracks = len(active_tracks)
    cv2.putText(display, f"Mot: {motion_pct:.1f}%  Trk: {n_tracks}",
                (4, y), font, 0.35, gray, 1)

    return display


# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════

class JSONLLogger:
    def __init__(self, path):
        self._file = open(path, "w")

    def log_track(self, track, cam_id):
        event = {
            "timestamp": datetime.now().isoformat(),
            "track_id": track.id,
            "first_frame": track.first_frame,
            "camera": cam_id,
            "kind": "known" if track.label else "unknown",
            "bbox": {"x": track.bbox[0], "y": track.bbox[1],
                     "w": track.bbox[2], "h": track.bbox[3]},
            "class": track.label or "unknown_anomaly",
            "confidence": track.confidence,
            "ae_error": round(track.score, 4),
            "detections": [
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


class FrameCSVLogger:
    """Per-frame CSV log with all pipeline signals."""

    HEADER = ("timestamp,frame,camera,motion_pct,ae_error,ae_threshold,"
              "anomaly,yolo_triggered,label,confidence,tracks,latency_ms\n")

    def __init__(self, path):
        self._file = open(path, "w")
        self._file.write(self.HEADER)

    def log(self, frame_num, cam_id, motion_pct, ae_error, ae_threshold,
            anomaly, yolo_triggered, label, confidence, n_tracks, latency_ms):
        ts = datetime.now().isoformat(timespec='milliseconds')
        ae_str = f"{ae_error:.4f}" if ae_error is not None else ""
        thresh_str = f"{ae_threshold:.4f}" if ae_threshold else ""
        lbl_str = label or ""
        conf_str = f"{confidence:.3f}" if confidence else ""
        self._file.write(
            f"{ts},{frame_num},{cam_id},{motion_pct:.2f},"
            f"{ae_str},{thresh_str},"
            f"{anomaly},{yolo_triggered},"
            f"{lbl_str},{conf_str},"
            f"{n_tracks},{latency_ms:.1f}\n"
        )

    def close(self):
        self._file.close()


class VideoWriterWrapper:
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
            self._writers[cam_id] = cv2.VideoWriter(
                path, fourcc, self.fps, (self.width, self.height))
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
    print("║   AE + YOLO · NCNN · Real-time       ║")
    print("╚══════════════════════════════════════╝\n")

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "detections.jsonl")
    csv_path = os.path.join(cfg.output_dir, "frames.csv")

    # ── Init ──
    camera = MultiCamera(cfg)
    src_fps = camera.src_fps or 30.0
    frame_budget_ms = 1000.0 / src_fps

    from classifier_ncnn import YOLOClassifier
    yolo = YOLOClassifier(cfg)
    yolo.start_async()

    ae = None
    if cfg.ae_enabled:
        from autoencoder_tflite import AutoencoderDetector
        ae = AutoencoderDetector(cfg)
        if ae.session is None:
            ae = None

    motion_dets = [MotionDetector(cfg) for _ in cfg.sources]
    trackers = [SimpleTracker(iou_thresh=cfg.tracker_iou,
                              max_lost=cfg.tracker_max_lost)
                for _ in cfg.sources]
    logger = JSONLLogger(log_path)
    frame_logger = FrameCSVLogger(csv_path)
    vid_writer = VideoWriterWrapper(
        cfg.output_dir, cfg.proc_width, cfg.proc_height) if cfg.save_video else None

    roi_y_start = int(cfg.proc_height * cfg.roi_top)
    roi_y_end = int(cfg.proc_height * cfg.roi_bottom)
    roi_h = roi_y_end - roi_y_start
    roi_area = roi_h * cfg.proc_width

    do_profile = cfg.profile
    skip_n = cfg.skip_frames
    ae_recheck = cfg.ae_recheck_interval

    stage_times = defaultdict(float)
    stage_counts = defaultdict(int)
    frame_count = 0
    processed_count = 0
    yolo_calls = 0
    ae_calls = 0
    ae_filtered = 0
    known_total = 0
    unknown_total = 0
    fps = 0.0
    t_start = time.time()
    t_processing = 0.0
    last_ae_error = None
    load_avg_ms = 0.0
    yolo_cooldown = 0              # frames remaining before next YOLO allowed
    yolo_cooldown_frames = 5       # min frames between YOLO submits

    # ── Print config ──
    ae_str = (f"ON ({cfg.ae_input_size}x{cfg.ae_input_size}, "
              f"thresh={cfg.ae_threshold}, "
              f"smooth={cfg.ae_smooth_min_hits}/{cfg.ae_smooth_window}, "
              f"recheck={ae_recheck})") if ae else "OFF"
    print(f"[Config] {cfg.proc_width}x{cfg.proc_height} | "
          f"ROI: {cfg.roi_top*100:.0f}%-{cfg.roi_bottom*100:.0f}% ({roi_h}px) | "
          f"skip={skip_n}")
    print(f"[Config] YOLO conf={cfg.yolo_conf} | "
          f"Tracker IoU={cfg.tracker_iou} max_lost={cfg.tracker_max_lost}")
    print(f"[Config] AE={ae_str}")
    print(f"[Config] Budget: {frame_budget_ms:.0f}ms/frame ({src_fps:.0f} FPS target)")
    print(f"[Config] Log → {log_path}")
    print("[Pipeline] Running... Press 'q' to quit.\n")

    gc.disable()

    try:
        while True:
            frames = camera.read_all()
            if all(f is None for f in frames):
                print("[Pipeline] All sources exhausted.")
                break

            frame_count += 1

            if skip_n > 0 and (frame_count % skip_n != 0):
                continue

            for idx, frame in enumerate(frames):
                if frame is None:
                    continue

                t_frame = time.time()
                roi = frame[roi_y_start:roi_y_end]
                processed_count += 1

                # ── Stage 1: Motion ──
                _t = time.time() if do_profile else 0
                mask, motion_regions = motion_dets[idx].detect(roi)
                if do_profile:
                    stage_times['motion'] += time.time() - _t
                    stage_counts['motion'] += 1

                # ── Pick up async YOLO results ──
                yolo_result = yolo.fetch()
                det_candidates = []
                if yolo_result is not None:
                    for det in yolo_result:
                        bx, by, bw, bh = det["bbox"]
                        x1, y1 = max(0, bx), max(0, by)
                        x2 = min(roi.shape[1], bx + bw)
                        y2 = min(roi.shape[0], by + bh)
                        crop = roi[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None
                        det_candidates.append({
                            "bbox": (bx, by, bw, bh),
                            "score": det["confidence"],
                            "image": crop,
                            "label": det["class_name"],
                            "confidence": det["confidence"],
                        })

                # ── Stage 2: Motion weighting ──
                motion_pct = 0.0
                if motion_regions:
                    total_motion = sum(a for _, a in motion_regions)
                    motion_pct = (total_motion / roi_area) * 100.0

                if motion_pct < 0.1:
                    motion_weight = 3.0
                elif motion_pct < 2.0:
                    motion_weight = 1.5
                elif motion_pct > 50.0:
                    motion_weight = 1.8
                elif motion_pct > 30.0:
                    motion_weight = 1.3
                else:
                    motion_weight = 1.0

                # ── Tracker coverage (only gates YOLO, not AE) ──
                tracker = trackers[idx]
                all_tracked = True
                if motion_regions:
                    for (mx, my, mw, mh), _ in motion_regions:
                        if not any(_iou((mx, my, mw, mh), tb) > 0.3
                                   for tb in tracker.active_bboxes):
                            all_tracked = False
                            break

                # ── Stage 3: AE — runs independently of tracker ──
                # AE always checks when there's motion (tracked or not)
                # Periodic re-check of tracked regions every N frames
                run_ae = False
                if ae is not None and motion_regions:
                    if not all_tracked:
                        run_ae = True
                    elif processed_count % ae_recheck == 0:
                        run_ae = True  # re-validate tracked regions

                    # Backpressure: reduce AE freq when CPU overloaded
                    if run_ae and load_avg_ms > frame_budget_ms * 1.5:
                        if processed_count % 2 != 0:
                            run_ae = False  # skip every other AE under load

                ae_says_anomaly = False
                if run_ae:
                    _t = time.time() if do_profile else 0
                    effective_threshold = cfg.ae_threshold * motion_weight
                    ae_says_anomaly, ae_error, hits = ae.is_anomaly_smoothed(
                        roi, effective_threshold)
                    ae_calls += 1
                    last_ae_error = ae_error
                    if do_profile:
                        stage_times['ae'] += time.time() - _t
                        stage_counts['ae'] += 1
                    if not ae_says_anomaly:
                        ae_filtered += 1

                # ── Stage 4: Submit YOLO (with cooldown) ──
                # YOLO gated by: tracker + AE + cooldown
                submit_yolo = False
                if not all_tracked:
                    if ae is not None:
                        submit_yolo = ae_says_anomaly
                    else:
                        submit_yolo = bool(motion_regions)
                elif ae_says_anomaly and processed_count % ae_recheck == 0:
                    submit_yolo = True

                # Cooldown: prevent rapid-fire YOLO when anomalies spike
                if yolo_cooldown > 0:
                    yolo_cooldown -= 1
                    submit_yolo = False

                if submit_yolo:
                    yolo.submit(roi)
                    yolo_calls += 1
                    yolo_cooldown = yolo_cooldown_frames  # reset cooldown

                # ── Stage 5: Tracker ──
                _t = time.time() if do_profile else 0
                new_tracks, active_tracks, gone_tracks = tracker.update(
                    det_candidates, frame_count)
                if do_profile:
                    stage_times['tracker'] += time.time() - _t
                    stage_counts['tracker'] += 1

                # ── Stage 6: Label + log ──
                for t in new_tracks:
                    for dc in det_candidates:
                        if dc["bbox"] == t.bbox:
                            t.label = dc["label"]
                            t.confidence = dc["confidence"]
                            t.yolo_dets = [{
                                "bbox": dc["bbox"],
                                "class_name": dc["label"],
                                "confidence": dc["confidence"],
                            }]
                            break
                    t.logged = True
                    logger.log_track(t, idx)
                    if t.label:
                        known_total += 1
                    else:
                        unknown_total += 1
                    t.image = None
                    t.yolo_dets = []

                for t in gone_tracks:
                    t.image = None
                det_candidates_for_log = det_candidates  # save before clearing
                det_candidates = None

                # ── Timing + backpressure ──
                elapsed = time.time() - t_frame
                elapsed_ms = elapsed * 1000
                t_processing += elapsed
                fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))
                load_avg_ms = 0.9 * load_avg_ms + 0.1 * elapsed_ms

                # ── Per-frame CSV log ──
                top_label = None
                top_conf = None
                if det_candidates_for_log:
                    best = max(det_candidates_for_log, key=lambda d: d['confidence'])
                    top_label = best['label']
                    top_conf = best['confidence']
                frame_logger.log(
                    frame_num=frame_count,
                    cam_id=idx,
                    motion_pct=motion_pct,
                    ae_error=last_ae_error,
                    ae_threshold=cfg.ae_threshold * motion_weight if ae else None,
                    anomaly=ae_says_anomaly,
                    yolo_triggered=submit_yolo,
                    label=top_label,
                    confidence=top_conf,
                    n_tracks=len(active_tracks),
                    latency_ms=elapsed_ms,
                )

                # ── Draw ──
                if cfg.save_video or cfg.show_preview:
                    _t = time.time() if do_profile else 0
                    hud = {
                        'ae_error': last_ae_error,
                        'ae_threshold': cfg.ae_threshold * motion_weight if ae else 0,
                        'ae_active': run_ae,
                        'yolo_busy': yolo.is_busy,
                        'yolo_cooldown': yolo_cooldown,
                        'yolo_calls': yolo_calls,
                        'load_ms': load_avg_ms,
                        'motion_pct': motion_pct,
                    }
                    display = draw_results(frame, roi_y_start, active_tracks,
                                           fps, motion_regions, hud)
                    if vid_writer:
                        vid_writer.write(idx, display)
                    if cfg.show_preview:
                        cv2.imshow(f"Camera {idx}", display)
                    if do_profile:
                        stage_times['draw'] += time.time() - _t
                        stage_counts['draw'] += 1

            if frame_count % 100 == 0:
                gc.collect()

            if frame_count % 30 == 0:
                total_tracks = sum(len(tr.tracks) for tr in trackers)
                ae_s = f" | AE: {last_ae_error:.3f}" if last_ae_error is not None else ""
                print(f"  Frame {frame_count} | FPS: {fps:.1f} | "
                      f"Tracks: {total_tracks} | YOLO: {yolo_calls} | "
                      f"load: {load_avg_ms:.0f}ms{ae_s}")

            if cfg.show_preview:
                wait_ms = max(1, int(1000 / src_fps) - int((time.time() - t_frame) * 1000)) if cfg.realtime else 1
                if cv2.waitKey(wait_ms) in (ord("q"), 27):
                    print("\n[Pipeline] Quit requested")
                    break

    finally:
        gc.enable()
        yolo.stop_async()
        camera.release()
        if vid_writer:
            vid_writer.release()
        frame_logger.close()
        logger.close()
        cv2.destroyAllWindows()

    # ── Summary ──
    elapsed_total = time.time() - t_start
    proc_fps = processed_count / max(t_processing, 0.001)
    print(f"\n{'='*55}")
    print(f"  Pipeline Complete!")
    print(f"{'='*55}")
    print(f"  Total frames: {frame_count}")
    print(f"  Processed:    {processed_count}")
    print(f"  Known:        {known_total}")
    print(f"  Unknown:      {unknown_total}")
    print(f"  YOLO calls:   {yolo_calls} / {processed_count} processed")
    if ae is not None:
        print(f"  AE calls:     {ae_calls} ({ae_filtered} filtered)")
    print(f"  Proc FPS:     {proc_fps:.1f}")
    print(f"  Wall time:    {elapsed_total:.1f}s")
    print(f"{'='*55}")

    if do_profile and stage_counts:
        print(f"\n  Per-stage timing (avg ms):")
        for s in ['motion', 'ae', 'yolo', 'tracker', 'draw']:
            if stage_counts.get(s, 0) > 0:
                avg = stage_times[s] / stage_counts[s] * 1000
                print(f"    {s:10s}: {avg:7.1f} ms  ({stage_counts[s]} calls)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Road Anomaly Detection Pipeline")
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--skip-frames", type=int, default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--proc-width", type=int, default=None)
    parser.add_argument("--proc-height", type=int, default=None)
    parser.add_argument("--roi-top", type=float, default=None)
    parser.add_argument("--tracker-iou", type=float, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--ae", action="store_true", help="Enable autoencoder")
    parser.add_argument("--no-ae", action="store_true", help="Disable autoencoder")
    parser.add_argument("--ae-threshold", type=float, default=None)
    parser.add_argument("--rpi", action="store_true",
                        help="RPi 4B preset: 320x240, AE+YOLO, no video/preview")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    cfg = Config.rpi_preset() if args.rpi else Config()
    if args.sources:
        cfg.sources = [int(s) if s.isdigit() else s for s in args.sources]
    if args.no_preview or args.headless:
        cfg.show_preview = False
    if args.yolo_conf is not None:
        cfg.yolo_conf = args.yolo_conf
    if args.output:
        cfg.output_dir = args.output
    if args.skip_frames is not None:
        cfg.skip_frames = args.skip_frames
    if args.realtime:
        cfg.realtime = True
    if args.proc_width is not None:
        cfg.proc_width = args.proc_width
    if args.proc_height is not None:
        cfg.proc_height = args.proc_height
    if args.roi_top is not None:
        cfg.roi_top = args.roi_top
    if args.tracker_iou is not None:
        cfg.tracker_iou = args.tracker_iou
    if args.no_video:
        cfg.save_video = False
    if args.ae:
        cfg.ae_enabled = True
    if args.no_ae:
        cfg.ae_enabled = False
    if args.ae_threshold is not None:
        cfg.ae_threshold = args.ae_threshold
    if args.profile:
        cfg.profile = True

    run(cfg)


if __name__ == "__main__":
    main()
