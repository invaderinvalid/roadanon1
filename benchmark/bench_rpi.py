#!/usr/bin/env python3
"""
RPi 4B Benchmark — tests all videos in test_video/ with different backends/settings.

Logs: FPS, CPU temperature, CPU usage, RAM usage per frame batch.
Generates separate report files per test configuration.

Usage (on RPi):
    python benchmark/bench_rpi.py                           # all defaults
    python benchmark/bench_rpi.py --backends ncnn tflite    # specific backends
    python benchmark/bench_rpi.py --vulkan                  # NCNN with Vulkan
    python benchmark/bench_rpi.py --video-dir test_video    # custom video dir
"""

import os
import sys
import json
import time
import glob
import platform
import argparse
import subprocess
from datetime import datetime

# Add parent dir so we can import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
# System monitors
# ═══════════════════════════════════════════════════════════════

def get_cpu_temp():
    """Read CPU temperature (RPi thermal zone)."""
    # RPi: /sys/class/thermal/thermal_zone0/temp
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except (FileNotFoundError, PermissionError):
        pass
    return -1.0  # not available (macOS etc.)


def get_cpu_usage():
    """Get CPU usage percent."""
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except ImportError:
        pass
    # Fallback: /proc/stat on Linux
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
    # Fallback: /proc/meminfo on Linux
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
# Pipeline stage timing
# ═══════════════════════════════════════════════════════════════

def run_benchmark(video_path, backend, settings, report_path, use_vulkan=False):
    """Run pipeline on a single video, logging performance metrics."""

    from config import Config
    from preprocessing import VideoSource, preprocess
    from motion_detect import MotionDetector
    from autoencoder import AnomalyDetector
    from cropping import Cropper, Validator

    cfg = Config()
    cfg.classifier_backend = backend
    cfg.show_preview = False
    cfg.sources = [video_path]

    # Apply settings overrides
    for k, v in settings.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Load modules
    src = VideoSource(video_path)
    motion = MotionDetector(cfg)
    ae = AnomalyDetector(cfg)
    cropper = Cropper(cfg)
    validator = Validator(cfg)

    # Classifier
    yolo = None
    if backend == "ncnn":
        os.environ["NCNN_VULKAN"] = "1" if use_vulkan else "0"
        from classifier_ncnn import YOLOClassifierNCNN
        yolo = YOLOClassifierNCNN(cfg)
    elif backend == "torch":
        from classifier_torch import YOLOClassifierTorch
        yolo = YOLOClassifierTorch(cfg)
    elif backend == "tflite":
        from classifier import YOLOClassifier
        yolo = YOLOClassifier(cfg)

    video_name = os.path.basename(video_path)
    print(f"\n  [{backend}{'_vulkan' if use_vulkan else ''}] {video_name}")
    print(f"  Resolution: {cfg.proc_width}x{cfg.proc_height} | "
          f"AE threshold: {cfg.ae_threshold} | Skip: {cfg.skip_frames}")

    # Per-frame log entries
    entries = []
    frame_num = 0
    t_start = time.time()

    while True:
        raw = src.read()
        if raw is None:
            break
        frame_num += 1
        frame = preprocess(raw, cfg.proc_width, cfg.proc_height)

        t0 = time.time()

        # Stage timings
        t_motion_s = time.time()
        mask, regions = motion.detect(frame)
        t_motion = time.time() - t_motion_s

        t_ae = 0.0
        t_crop = 0.0
        t_yolo = 0.0
        num_crops = 0
        num_dets = 0
        mean_error = 0.0

        if regions:
            # Mask frame
            masked = frame.copy()
            masked[mask > 0] = 0

            t_ae_s = time.time()
            error_map, mean_error = ae.infer(masked)
            error_map[mask > 0] = 0.0
            unmasked = mask == 0
            if unmasked.any():
                mean_error = float(error_map[unmasked].mean())
            t_ae = time.time() - t_ae_s

            if mean_error > cfg.ae_threshold:
                t_crop_s = time.time()
                crops = cropper.extract(frame, error_map, cfg.ae_threshold)
                t_crop = time.time() - t_crop_s

                t_yolo_s = time.time()
                yolo_count = 0
                for crop in crops:
                    if yolo_count >= cfg.max_crops:
                        break
                    if yolo is not None and validator.is_valid(crop["image"]):
                        dets = yolo.infer(crop["image"])
                        num_dets += len(dets)
                        yolo_count += 1
                num_crops = yolo_count
                t_yolo = time.time() - t_yolo_s

        t_total = time.time() - t0

        # Sample system stats every 10 frames
        entry = {
            "frame": frame_num,
            "t_total_ms": round(t_total * 1000, 2),
            "t_motion_ms": round(t_motion * 1000, 2),
            "t_ae_ms": round(t_ae * 1000, 2),
            "t_crop_ms": round(t_crop * 1000, 2),
            "t_yolo_ms": round(t_yolo * 1000, 2),
            "ae_error": round(mean_error, 4),
            "num_crops": num_crops,
            "num_dets": num_dets,
            "fps_inst": round(1.0 / max(t_total, 0.001), 1),
        }

        if frame_num % 10 == 0:
            entry["cpu_temp"] = get_cpu_temp()
            entry["cpu_percent"] = get_cpu_usage()
            entry["ram"] = get_ram_usage()

        entries.append(entry)

        if frame_num % 50 == 0:
            fps_avg = frame_num / (time.time() - t_start)
            print(f"    Frame {frame_num} | FPS: {fps_avg:.1f} | "
                  f"Temp: {entry.get('cpu_temp', '?')}°C")

    src.release()
    elapsed = time.time() - t_start

    # Summary
    fps_avg = frame_num / max(elapsed, 0.001)
    avg_total = np.mean([e["t_total_ms"] for e in entries])
    avg_motion = np.mean([e["t_motion_ms"] for e in entries])
    avg_ae = np.mean([e["t_ae_ms"] for e in entries if e["t_ae_ms"] > 0])
    avg_yolo = np.mean([e["t_yolo_ms"] for e in entries if e["t_yolo_ms"] > 0]) if any(e["t_yolo_ms"] > 0 for e in entries) else 0

    summary = {
        "video": video_name,
        "backend": backend,
        "vulkan": use_vulkan,
        "resolution": f"{cfg.proc_width}x{cfg.proc_height}",
        "ae_threshold": cfg.ae_threshold,
        "skip_frames": cfg.skip_frames,
        "max_crops": cfg.max_crops,
        "total_frames": frame_num,
        "elapsed_s": round(elapsed, 2),
        "avg_fps": round(fps_avg, 1),
        "avg_total_ms": round(avg_total, 2),
        "avg_motion_ms": round(avg_motion, 2),
        "avg_ae_ms": round(float(avg_ae), 2) if not np.isnan(avg_ae) else 0,
        "avg_yolo_ms": round(float(avg_yolo), 2),
        "platform": platform.machine(),
        "python": platform.python_version(),
    }

    print(f"  ✓ {frame_num} frames | {fps_avg:.1f} FPS | {elapsed:.1f}s")
    print(f"    Avg: total={avg_total:.1f}ms motion={avg_motion:.1f}ms "
          f"ae={float(avg_ae):.1f}ms yolo={float(avg_yolo):.1f}ms")

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
    parser.add_argument("--video-dir", default="test_video",
                        help="Directory containing test videos")
    parser.add_argument("--backends", nargs="+", default=["torch"],
                        choices=["ncnn", "torch", "tflite"],
                        help="Classifier backends to test")
    parser.add_argument("--vulkan", action="store_true",
                        help="Also test NCNN with Vulkan acceleration")
    parser.add_argument("--ae-threshold", type=float, default=0.07)
    parser.add_argument("--ae-backend", choices=["torch", "onnx"], default="onnx",
                        help="Autoencoder backend (onnx recommended for RPi 4B)")
    parser.add_argument("--skip-frames", type=int, default=0)
    parser.add_argument("--max-crops", type=int, default=5)
    parser.add_argument("--proc-width", type=int, default=640)
    parser.add_argument("--proc-height", type=int, default=480)
    parser.add_argument("--output-dir", default="benchmark/reports",
                        help="Where to write report logs")
    args = parser.parse_args()

    # Find test videos
    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if not videos:
        print(f"No .mp4 files found in {args.video_dir}/")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    settings = {
        "ae_threshold": args.ae_threshold,
        "ae_backend": args.ae_backend,
        "skip_frames": args.skip_frames,
        "max_crops": args.max_crops,
        "proc_width": args.proc_width,
        "proc_height": args.proc_height,
    }

    print("╔══════════════════════════════════════╗")
    print("║   RPi 4B Pipeline Benchmark          ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Videos: {len(videos)} | Backends: {args.backends} | AE: {args.ae_backend}")
    print(f"  Vulkan: {args.vulkan} | Resolution: {args.proc_width}x{args.proc_height}")
    print(f"  Platform: {platform.machine()} / {platform.system()}")

    all_summaries = []

    for backend in args.backends:
        # Standard run
        for video in videos:
            vname = os.path.splitext(os.path.basename(video))[0]
            report_path = os.path.join(
                args.output_dir, f"{timestamp}_{backend}_{vname}.json")
            summary = run_benchmark(video, backend, settings, report_path)
            all_summaries.append(summary)

        # Vulkan run (NCNN only)
        if args.vulkan and backend == "ncnn":
            for video in videos:
                vname = os.path.splitext(os.path.basename(video))[0]
                report_path = os.path.join(
                    args.output_dir, f"{timestamp}_{backend}_vulkan_{vname}.json")
                summary = run_benchmark(video, backend, settings, report_path,
                                        use_vulkan=True)
                all_summaries.append(summary)

    # Write combined summary
    combined_path = os.path.join(args.output_dir, f"{timestamp}_summary.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Video':<25} {'Backend':<15} {'FPS':>6} {'Avg ms':>8} {'AE ms':>7} {'YOLO ms':>8}")
    print(f"  {'-'*25} {'-'*15} {'-'*6} {'-'*8} {'-'*7} {'-'*8}")
    for s in all_summaries:
        bk = s["backend"] + ("_vulkan" if s.get("vulkan") else "")
        print(f"  {s['video']:<25} {bk:<15} {s['avg_fps']:>6.1f} "
              f"{s['avg_total_ms']:>8.1f} {s['avg_ae_ms']:>7.1f} {s['avg_yolo_ms']:>8.1f}")
    print(f"{'='*70}")
    print(f"  Combined → {combined_path}")


if __name__ == "__main__":
    main()
