#!/usr/bin/env python3
"""
RPi 4B Benchmark — mirrors the actual pipeline.py flow exactly.

Stages (same as pipeline.py):
  1. Read frame → crop to ROI
  2. Motion mask on ROI
  3. Motion gate (skip AE if motion too small/large)
  4. Mask → AE on masked ROI
  5. Error map → crop from original ROI
  6. Tracker update (IoU matching)
  7. YOLO on NEW tracks only
  8. Log

Logs: per-stage timing, CPU temp, CPU usage, RAM.
Supports --rpi preset (same as pipeline.py --rpi).

Usage (on RPi):
    python benchmark/bench_rpi.py --rpi --sources test_video/*.mp4
    python benchmark/bench_rpi.py --rpi --backends ncnn tflite
    python benchmark/bench_rpi.py --sources test_video/66_10-07-2023.mp4
"""

import os
import sys
import json
import time
import glob
import platform
import argparse
from datetime import datetime
from collections import defaultdict

# Add parent dir so we can import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
# System monitors
# ═══════════════════════════════════════════════════════════════

def get_cpu_temp():
    """Read CPU temperature (RPi thermal zone)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except (FileNotFoundError, PermissionError):
        return -1.0


def get_cpu_usage():
    """Get CPU usage percent."""
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except ImportError:
        pass
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        idle = int(parts[4])
        total = sum(int(x) for x in parts[1:])
        return round((1.0 - idle / total) * 100, 1)
    except Exception:
        return -1.0


def get_ram_usage():
    """Get RAM usage in MB and percent."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {"used_mb": round(mem.used / 1024 / 1024, 1),
                "total_mb": round(mem.total / 1024 / 1024, 1),
                "percent": mem.percent}
    except ImportError:
        pass
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k.strip()] = int(v.strip().split()[0])
        total = info["MemTotal"] / 1024
        avail = info.get("MemAvailable", info.get("MemFree", 0)) / 1024
        used = total - avail
        return {"used_mb": round(used, 1), "total_mb": round(total, 1),
                "percent": round(used / total * 100, 1)}
    except Exception:
        return {"used_mb": -1, "total_mb": -1, "percent": -1}


# ═══════════════════════════════════════════════════════════════
# Benchmark run — mirrors pipeline.py run() exactly
# ═══════════════════════════════════════════════════════════════

def run_benchmark(video_path, cfg, report_path):
    """Run the actual pipeline flow on a single video, logging perf metrics."""

    from preprocessing import VideoSource
    from motion_detect import MotionDetector
    from autoencoder import AnomalyDetector
    from cropping import Cropper, Validator
    from tracker import SimpleTracker

    # Load classifier
    if cfg.classifier_backend == "ncnn":
        from classifier_ncnn import YOLOClassifierNCNN
        yolo = YOLOClassifierNCNN(cfg)
    elif cfg.classifier_backend == "torch":
        from classifier_torch import YOLOClassifierTorch
        yolo = YOLOClassifierTorch(cfg)
    else:
        from classifier import YOLOClassifier
        yolo = YOLOClassifier(cfg)

    # Open video
    src = VideoSource(video_path)

    # ROI: vertical slice (same as pipeline.py)
    roi_y_start = int(cfg.proc_height * cfg.roi_top)
    roi_y_end = int(cfg.proc_height * cfg.roi_bottom)
    roi_h = roi_y_end - roi_y_start
    roi_area = roi_h * cfg.proc_width

    # Motion gate thresholds (same as pipeline.py)
    motion_min_pct = getattr(cfg, 'motion_min_pct', 0.5)
    motion_max_pct = getattr(cfg, 'motion_max_pct', 60.0)

    # Modules (same as pipeline.py)
    motion_det = MotionDetector(cfg)
    ae = AnomalyDetector(cfg)
    cropper = Cropper(cfg)
    validator = Validator(cfg)
    tracker = SimpleTracker(iou_thresh=cfg.tracker_iou,
                            max_lost=cfg.tracker_max_lost)

    video_name = os.path.basename(video_path)
    backend_str = f"{cfg.classifier_backend}"
    ae_str = f"{cfg.ae_backend}" + ("-int8" if getattr(cfg, 'ae_use_int8', False) else "")

    print(f"\n  [{backend_str} / {ae_str}] {video_name}")
    print(f"  Resolution: {cfg.proc_width}x{cfg.proc_height} | "
          f"ROI: {cfg.roi_top*100:.0f}%-{cfg.roi_bottom*100:.0f}% ({roi_h}px)")
    print(f"  AE thresh: {cfg.ae_threshold} | YOLO conf: {cfg.yolo_conf} | "
          f"Skip: {cfg.skip_frames} | Max crops: {cfg.max_crops}")

    # Per-frame entries + stage accumulators
    entries = []
    stage_times = defaultdict(float)
    stage_counts = defaultdict(int)

    frame_num = 0
    processed = 0
    yolo_calls = 0
    known_total = 0
    unknown_total = 0
    ae_skipped = 0
    skip_n = cfg.skip_frames
    t_start = time.time()

    while True:
        raw = src.read()
        if raw is None:
            break
        frame_num += 1

        # Resize to proc dimensions
        frame = cv2.resize(raw, (cfg.proc_width, cfg.proc_height))

        # ── Frame skipping (same as pipeline.py) ──
        if skip_n > 0 and (frame_num % skip_n != 0):
            roi = frame[roi_y_start:roi_y_end]
            motion_det.detect(roi)  # keep BG model updated
            continue

        processed += 1
        t_frame = time.time()

        entry = {"frame": frame_num}
        mean_error = 0.0
        motion_regions = []
        det_candidates = []

        # ── Stage 1: Crop to ROI ──
        roi = frame[roi_y_start:roi_y_end]

        # ── Stage 2: Motion mask on ROI ──
        t0 = time.time()
        mask, motion_regions = motion_det.detect(roi)
        dt = time.time() - t0
        entry["t_motion_ms"] = round(dt * 1000, 2)
        stage_times['motion'] += dt
        stage_counts['motion'] += 1

        entry["t_ae_ms"] = 0.0
        entry["t_crop_ms"] = 0.0
        entry["t_yolo_ms"] = 0.0
        entry["t_tracker_ms"] = 0.0
        entry["num_crops"] = 0
        entry["num_dets"] = 0
        entry["ae_error"] = 0.0
        entry["ae_skipped"] = False
        entry["motion_pct"] = 0.0

        if motion_regions:
            # ── Motion gate (same as pipeline.py) ──
            total_motion = sum(a for _, a in motion_regions)
            motion_pct = (total_motion / roi_area) * 100.0
            entry["motion_pct"] = round(motion_pct, 2)

            if motion_pct < motion_min_pct or motion_pct > motion_max_pct:
                ae_skipped += 1
                entry["ae_skipped"] = True
            else:
                # ── Stage 3: Mask moving objects ──
                t0 = time.time()
                mask_bool = mask > 0
                masked_roi = roi.copy()
                masked_roi[mask_bool] = 0
                dt = time.time() - t0
                stage_times['mask'] += dt
                stage_counts['mask'] += 1

                # ── Stage 4: AE on masked ROI ──
                t0 = time.time()
                error_map, mean_error = ae.infer(masked_roi)
                error_map[mask_bool] = 0.0
                unmasked = ~mask_bool
                if unmasked.any():
                    mean_error = float(error_map[unmasked].mean())
                else:
                    mean_error = 0.0
                dt = time.time() - t0
                entry["t_ae_ms"] = round(dt * 1000, 2)
                entry["ae_error"] = round(mean_error, 4)
                stage_times['ae'] += dt
                stage_counts['ae'] += 1

                if mean_error > cfg.ae_threshold:
                    # ── Stage 5: Crop from ORIGINAL ROI ──
                    t0 = time.time()
                    crops = cropper.extract(roi, error_map, cfg.ae_threshold)
                    crop_count = 0
                    for crop in crops:
                        if crop_count >= cfg.max_crops:
                            break
                        if validator.is_valid(crop["image"]):
                            det_candidates.append({
                                "bbox": crop["bbox"],
                                "score": crop["score"],
                                "image": crop["image"],
                            })
                            crop_count += 1
                    dt = time.time() - t0
                    entry["t_crop_ms"] = round(dt * 1000, 2)
                    entry["num_crops"] = crop_count
                    stage_times['crop'] += dt
                    stage_counts['crop'] += 1

        # ── Stage 6: Tracker update ──
        t0 = time.time()
        new_tracks, active_tracks, gone_tracks = tracker.update(
            det_candidates, frame_num)
        dt = time.time() - t0
        entry["t_tracker_ms"] = round(dt * 1000, 2)
        stage_times['tracker'] += dt
        stage_counts['tracker'] += 1

        # ── Stage 7: YOLO on NEW tracks only ──
        t0 = time.time()
        for t in new_tracks:
            if t.image is not None and validator.is_valid(t.image):
                dets = yolo.infer(t.image)
                yolo_calls += 1
                t.yolo_dets = dets
                if dets:
                    t.label = dets[0]["class_name"]
                    t.confidence = dets[0]["confidence"]

            t.logged = True
            if t.label:
                known_total += 1
            else:
                unknown_total += 1
            entry["num_dets"] += len(t.yolo_dets)

        dt = time.time() - t0
        if new_tracks:
            entry["t_yolo_ms"] = round(dt * 1000, 2)
            stage_times['yolo'] += dt
            stage_counts['yolo'] += 1

        t_total = time.time() - t_frame
        entry["t_total_ms"] = round(t_total * 1000, 2)
        entry["fps_inst"] = round(1.0 / max(t_total, 0.001), 1)
        entry["active_tracks"] = len(active_tracks)

        # Sample system stats every 10 processed frames
        if processed % 10 == 0:
            entry["cpu_temp"] = get_cpu_temp()
            entry["cpu_percent"] = get_cpu_usage()
            entry["ram"] = get_ram_usage()

        entries.append(entry)

        if processed % 30 == 0:
            fps_avg = processed / (time.time() - t_start)
            temp = entry.get('cpu_temp', '?')
            print(f"    Frame {frame_num} (proc {processed}) | FPS: {fps_avg:.1f} | "
                  f"Tracks: {len(active_tracks)} | Temp: {temp}°C")

    src.release()
    elapsed = time.time() - t_start

    # ─── Summary ───
    fps_avg = processed / max(elapsed, 0.001)
    ae_entries = [e for e in entries if e["t_ae_ms"] > 0]

    summary = {
        "video": video_name,
        "yolo_backend": cfg.classifier_backend,
        "ae_backend": ae_str,
        "resolution": f"{cfg.proc_width}x{cfg.proc_height}",
        "roi": f"{cfg.roi_top*100:.0f}%-{cfg.roi_bottom*100:.0f}%",
        "roi_height": roi_h,
        "ae_threshold": cfg.ae_threshold,
        "yolo_conf": cfg.yolo_conf,
        "skip_frames": cfg.skip_frames,
        "max_crops": cfg.max_crops,
        "total_frames": frame_num,
        "processed_frames": processed,
        "elapsed_s": round(elapsed, 2),
        "avg_fps": round(fps_avg, 1),
        "avg_total_ms": round(np.mean([e["t_total_ms"] for e in entries]), 2),
        "avg_motion_ms": round(np.mean([e["t_motion_ms"] for e in entries]), 2),
        "avg_ae_ms": round(float(np.mean([e["t_ae_ms"] for e in ae_entries])), 2) if ae_entries else 0,
        "avg_crop_ms": round(float(np.mean([e["t_crop_ms"] for e in entries if e["t_crop_ms"] > 0])), 2) if any(e["t_crop_ms"] > 0 for e in entries) else 0,
        "avg_yolo_ms": round(float(np.mean([e["t_yolo_ms"] for e in entries if e["t_yolo_ms"] > 0])), 2) if any(e["t_yolo_ms"] > 0 for e in entries) else 0,
        "avg_tracker_ms": round(np.mean([e["t_tracker_ms"] for e in entries]), 2),
        "yolo_calls": yolo_calls,
        "known_anomalies": known_total,
        "unknown_anomalies": unknown_total,
        "ae_skipped_gate": ae_skipped,
        "platform": platform.machine(),
        "python": platform.python_version(),
    }

    print(f"  ✓ {frame_num} frames ({processed} processed) | "
          f"{fps_avg:.1f} FPS | {elapsed:.1f}s")
    print(f"    Avg: total={summary['avg_total_ms']:.1f}ms "
          f"motion={summary['avg_motion_ms']:.1f}ms "
          f"ae={summary['avg_ae_ms']:.1f}ms "
          f"yolo={summary['avg_yolo_ms']:.1f}ms "
          f"tracker={summary['avg_tracker_ms']:.1f}ms")
    print(f"    YOLO calls: {yolo_calls} | Known: {known_total} | "
          f"Unknown: {unknown_total} | AE skipped: {ae_skipped}")

    # Per-stage breakdown
    print(f"    Per-stage avg (ms):")
    for s in ['motion', 'mask', 'ae', 'crop', 'tracker', 'yolo']:
        if stage_counts.get(s, 0) > 0:
            avg = stage_times[s] / stage_counts[s] * 1000
            print(f"      {s:10s}: {avg:7.1f} ms  ({stage_counts[s]} calls)")

    # Write report
    report = {"summary": summary, "frames": entries}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Report → {report_path}")

    return summary


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RPi 4B Pipeline Benchmark")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Video files to benchmark (default: test_video/*.mp4)")
    parser.add_argument("--video-dir", default="test_video",
                        help="Directory containing test videos (used if --sources omitted)")
    parser.add_argument("--backends", nargs="+", default=["ncnn"],
                        choices=["ncnn", "torch", "tflite"],
                        help="YOLO backends to test")
    parser.add_argument("--ae-backend", choices=["torch", "onnx"], default=None,
                        help="AE backend override (default from preset)")
    parser.add_argument("--rpi", action="store_true",
                        help="Use RPi 4B preset (same as pipeline.py --rpi)")
    parser.add_argument("--ae-threshold", type=float, default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--skip-frames", type=int, default=None)
    parser.add_argument("--max-crops", type=int, default=None)
    parser.add_argument("--proc-width", type=int, default=None)
    parser.add_argument("--proc-height", type=int, default=None)
    parser.add_argument("--roi-top", type=float, default=None)
    parser.add_argument("--output-dir", default="benchmark/reports",
                        help="Where to write report JSON files")
    args = parser.parse_args()

    from config import Config

    # Build base config (same logic as pipeline.py)
    base_cfg = Config.rpi_preset() if args.rpi else Config()
    base_cfg.show_preview = False
    base_cfg.save_video = False

    # CLI overrides
    if args.ae_backend is not None:
        base_cfg.ae_backend = args.ae_backend
    if args.ae_threshold is not None:
        base_cfg.ae_threshold = args.ae_threshold
    if args.yolo_conf is not None:
        base_cfg.yolo_conf = args.yolo_conf
    if args.skip_frames is not None:
        base_cfg.skip_frames = args.skip_frames
    if args.max_crops is not None:
        base_cfg.max_crops = args.max_crops
    if args.proc_width is not None:
        base_cfg.proc_width = args.proc_width
    if args.proc_height is not None:
        base_cfg.proc_height = args.proc_height
    if args.roi_top is not None:
        base_cfg.roi_top = args.roi_top

    # Find videos
    if args.sources:
        videos = []
        for s in args.sources:
            videos.extend(glob.glob(s))
        videos = sorted(set(videos))
    else:
        videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))

    if not videos:
        print(f"No .mp4 files found")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ae_str = f"{base_cfg.ae_backend}" + ("-int8" if getattr(base_cfg, 'ae_use_int8', False) else "")

    print("╔══════════════════════════════════════╗")
    print("║   RPi 4B Pipeline Benchmark          ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Videos:     {len(videos)}")
    print(f"  YOLO:       {args.backends}")
    print(f"  AE:         {ae_str}")
    print(f"  Resolution: {base_cfg.proc_width}x{base_cfg.proc_height}")
    print(f"  ROI:        {base_cfg.roi_top*100:.0f}%-{base_cfg.roi_bottom*100:.0f}%")
    print(f"  Skip:       {base_cfg.skip_frames} | Max crops: {base_cfg.max_crops}")
    print(f"  AE thresh:  {base_cfg.ae_threshold} | YOLO conf: {base_cfg.yolo_conf}")
    print(f"  Platform:   {platform.machine()} / {platform.system()}")
    if args.rpi:
        print(f"  Preset:     --rpi")

    all_summaries = []

    for backend in args.backends:
        # Fresh config per backend (to avoid cross-contamination)
        cfg = Config.rpi_preset() if args.rpi else Config()
        cfg.show_preview = False
        cfg.save_video = False
        cfg.classifier_backend = backend

        # Apply same CLI overrides
        if args.ae_backend is not None:
            cfg.ae_backend = args.ae_backend
        if args.ae_threshold is not None:
            cfg.ae_threshold = args.ae_threshold
        if args.yolo_conf is not None:
            cfg.yolo_conf = args.yolo_conf
        if args.skip_frames is not None:
            cfg.skip_frames = args.skip_frames
        if args.max_crops is not None:
            cfg.max_crops = args.max_crops
        if args.proc_width is not None:
            cfg.proc_width = args.proc_width
        if args.proc_height is not None:
            cfg.proc_height = args.proc_height
        if args.roi_top is not None:
            cfg.roi_top = args.roi_top

        for video in videos:
            vname = os.path.splitext(os.path.basename(video))[0]
            report_path = os.path.join(
                args.output_dir, f"{timestamp}_{backend}_{vname}.json")
            summary = run_benchmark(video, cfg, report_path)
            all_summaries.append(summary)

    # Combined summary
    combined_path = os.path.join(args.output_dir, f"{timestamp}_summary.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Video':<25} {'YOLO':<8} {'AE':<12} {'FPS':>6} {'Total':>7} "
          f"{'AE':>6} {'YOLO':>6} {'Trk':>5} {'YOcall':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*6} {'-'*7} "
          f"{'-'*6} {'-'*6} {'-'*5} {'-'*6}")
    for s in all_summaries:
        print(f"  {s['video']:<25} {s['yolo_backend']:<8} {s['ae_backend']:<12} "
              f"{s['avg_fps']:>6.1f} {s['avg_total_ms']:>6.1f}ms "
              f"{s['avg_ae_ms']:>5.1f}ms {s['avg_yolo_ms']:>5.1f}ms "
              f"{s['avg_tracker_ms']:>4.1f}ms {s['yolo_calls']:>6d}")
    print(f"{'='*80}")
    print(f"  Combined → {combined_path}")


if __name__ == "__main__":
    main()
