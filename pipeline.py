"""Road Anomaly Detection Pipeline — Real-time on RPi 4B.

Async YOLO flow:
  Main thread:  capture → motion → tracker → draw  (runs at camera FPS)
  YOLO thread:  inference in background (non-blocking)

  Results from YOLO feed into tracker on the NEXT frame.
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


def draw_results(frame, roi_y, active_tracks, fps, motion_regions=None):
    """Draw bounding boxes, motion regions, and status on frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    if roi_y > 0:
        for x0 in range(0, w, 12):
            cv2.line(display, (x0, roi_y), (min(x0 + 6, w), roi_y), (255, 255, 0), 1)

    if motion_regions:
        for (mx, my, mw, mh), _ in motion_regions:
            cv2.rectangle(display, (mx, roi_y + my), (mx + mw, roi_y + my + mh),
                          (0, 180, 0), 1)

    for t in active_tracks:
        tx, ty, tw, th = t.bbox
        fy = roi_y + ty
        color = CLASS_COLORS.get(t.label, (255, 255, 0))
        thickness = 2 if t.label else 1
        cv2.rectangle(display, (tx, fy), (tx + tw, fy + th), color, thickness)
        lbl = f"T{t.id} {t.label} {t.confidence:.2f}" if t.label else f"T{t.id} ?"
        cv2.putText(display, lbl, (tx, max(fy - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, f"Tracks: {len(active_tracks)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
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
# Pipeline — async YOLO for real-time
# ═══════════════════════════════════════════════════════════════

def run(cfg: Config):
    print("╔══════════════════════════════════════╗")
    print("║   Road Anomaly Detection Pipeline    ║")
    print("║   Real-time · NCNN · Async YOLO      ║")
    print("╚══════════════════════════════════════╝\n")

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "detections.jsonl")

    # ── Init ──
    camera = MultiCamera(cfg)
    src_fps = camera.src_fps or 30.0

    from classifier_ncnn import YOLOClassifier
    yolo = YOLOClassifier(cfg)
    yolo.start_async()

    motion_dets = [MotionDetector(cfg) for _ in cfg.sources]
    trackers = [SimpleTracker(iou_thresh=cfg.tracker_iou,
                              max_lost=cfg.tracker_max_lost)
                for _ in cfg.sources]
    logger = JSONLLogger(log_path)
    vid_writer = VideoWriterWrapper(
        cfg.output_dir, cfg.proc_width, cfg.proc_height) if cfg.save_video else None

    roi_y_start = int(cfg.proc_height * cfg.roi_top)
    roi_y_end = int(cfg.proc_height * cfg.roi_bottom)
    roi_h = roi_y_end - roi_y_start
    roi_area = roi_h * cfg.proc_width

    motion_min_pct = cfg.motion_min_pct
    motion_max_pct = cfg.motion_max_pct
    do_profile = cfg.profile
    skip_n = cfg.skip_frames

    stage_times = defaultdict(float)
    stage_counts = defaultdict(int)
    frame_count = 0
    yolo_calls = 0
    known_total = 0
    unknown_total = 0
    motion_skipped = 0
    fps = 0.0
    t_start = time.time()

    # Pending YOLO results from async — applied on next frame
    pending_dets = [[] for _ in cfg.sources]

    print(f"[Config] {cfg.proc_width}x{cfg.proc_height} | "
          f"ROI: {cfg.roi_top*100:.0f}%-{cfg.roi_bottom*100:.0f}% ({roi_h}px) | "
          f"skip={skip_n}")
    print(f"[Config] YOLO conf={cfg.yolo_conf} | "
          f"Tracker IoU={cfg.tracker_iou} max_lost={cfg.tracker_max_lost}")
    print(f"[Config] Video={'ON' if cfg.save_video else 'OFF'} | "
          f"Preview={'ON' if cfg.show_preview else 'OFF'} | "
          f"Profile={'ON' if do_profile else 'OFF'}")
    print(f"[Config] Log → {log_path}")
    print("[Pipeline] Running (async YOLO)... Press 'q' to quit.\n")

    gc.disable()

    try:
        while True:
            frames = camera.read_all()
            if all(f is None for f in frames):
                print("[Pipeline] All sources exhausted.")
                break

            frame_count += 1

            # Skip frames — keep BG model fresh
            if skip_n > 0 and (frame_count % skip_n != 0):
                for idx, frame in enumerate(frames):
                    if frame is not None:
                        motion_dets[idx].detect(frame[roi_y_start:roi_y_end])
                continue

            for idx, frame in enumerate(frames):
                if frame is None:
                    continue

                t_frame = time.time()
                roi = frame[roi_y_start:roi_y_end]

                # ── Motion detection ──
                _t = time.time() if do_profile else 0
                mask, motion_regions = motion_dets[idx].detect(roi)
                if do_profile:
                    stage_times['motion'] += time.time() - _t
                    stage_counts['motion'] += 1

                # ── Pick up async YOLO results from previous frame ──
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

                # ── Submit new YOLO request (non-blocking) ──
                if motion_regions:
                    total_motion = sum(a for _, a in motion_regions)
                    motion_pct = (total_motion / roi_area) * 100.0

                    if motion_pct < motion_min_pct or motion_pct > motion_max_pct:
                        motion_skipped += 1
                    else:
                        tracker = trackers[idx]
                        has_untracked = False
                        for (mx, my, mw, mh), _ in motion_regions:
                            if not any(_iou((mx, my, mw, mh), tb) > 0.3
                                       for tb in tracker.active_bboxes):
                                has_untracked = True
                                break

                        if has_untracked and not yolo.is_busy:
                            if yolo.submit(roi):
                                yolo_calls += 1

                # ── Tracker update ──
                _t = time.time() if do_profile else 0
                tracker = trackers[idx]
                new_tracks, active_tracks, gone_tracks = tracker.update(
                    det_candidates, frame_count)
                if do_profile:
                    stage_times['tracker'] += time.time() - _t
                    stage_counts['tracker'] += 1

                # ── Label + log new tracks ──
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
                det_candidates = None

                # ── Draw ──
                elapsed = time.time() - t_frame
                fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))

                if cfg.save_video or cfg.show_preview:
                    _t = time.time() if do_profile else 0
                    display = draw_results(frame, roi_y_start, active_tracks,
                                           fps, motion_regions)
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
                print(f"  Frame {frame_count} | FPS: {fps:.1f} | "
                      f"Tracks: {total_tracks} | YOLO: {yolo_calls}")

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
        logger.close()
        cv2.destroyAllWindows()

    elapsed_total = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Pipeline Complete!")
    print(f"{'='*55}")
    print(f"  Frames:       {frame_count}")
    print(f"  Known:        {known_total}")
    print(f"  Unknown:      {unknown_total}")
    print(f"  YOLO calls:   {yolo_calls} / {frame_count} frames")
    print(f"  Avg FPS:      {frame_count / max(elapsed_total, 0.001):.1f}")
    print(f"  Time:         {elapsed_total:.1f}s")
    print(f"  Skipped:      {motion_skipped} (motion gate)")
    print(f"{'='*55}")

    if do_profile and stage_counts:
        print(f"\n  Per-stage timing (avg ms):")
        for s in ['motion', 'yolo', 'tracker', 'draw']:
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
    parser.add_argument("--rpi", action="store_true",
                        help="RPi 4B preset: 320x240, skip=4, no video/preview")
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
    if args.profile:
        cfg.profile = True

    run(cfg)


if __name__ == "__main__":
    main()
