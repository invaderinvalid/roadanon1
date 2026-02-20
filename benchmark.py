#!/usr/bin/env python3
"""Benchmark — 10-minute stress test with hardware monitoring.

Loops test videos continuously while logging:
  - CPU % (per-core + total)
  - RAM usage (MB + %)
  - CPU temperature (RPi / Mac)
  - Thread count
  - Pipeline FPS + latency
  - AE calls, YOLO calls

Output:
  output/benchmark.csv      — per-second samples
  output/benchmark_summary.txt — final report

Usage:
  python benchmark.py                     # 10 min default
  python benchmark.py --duration 300      # 5 min
  python benchmark.py --duration 60       # 1 min quick test
"""

import os
import sys
import time
import glob
import csv
import threading
import argparse
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import psutil

from config import Config
from motion_detect import MotionDetector
from tracker import SimpleTracker, _iou


# ═══════════════════════════════════════════════════════════════
# System Monitor
# ═══════════════════════════════════════════════════════════════

def get_cpu_temp():
    """Get CPU temperature (works on RPi and Mac)."""
    # RPi — thermal zone
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip()) / 1000.0
    except (FileNotFoundError, PermissionError):
        pass

    # macOS — via psutil (may not work on all systems)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries:
                    return entries[0].current
    except (AttributeError, Exception):
        pass

    # macOS fallback — powermetrics (too slow for real-time, skip)
    return None


class SystemMonitor:
    """Samples system stats at regular intervals."""

    def __init__(self, sample_interval=1.0):
        self.interval = sample_interval
        self.samples = []
        self._running = False
        self._thread = None
        self._process = psutil.Process(os.getpid())
        # Prime CPU percent
        self._process.cpu_percent()
        psutil.cpu_percent(percpu=True)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self._running:
            sample = self._sample()
            self.samples.append(sample)
            time.sleep(self.interval)

    def _sample(self):
        mem = psutil.virtual_memory()
        proc_mem = self._process.memory_info()
        cpu_per_core = psutil.cpu_percent(percpu=True)
        temp = get_cpu_temp()

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": time.time(),
            "cpu_total_pct": sum(cpu_per_core) / len(cpu_per_core),
            "cpu_max_core_pct": max(cpu_per_core),
            "cpu_cores": len(cpu_per_core),
            "cpu_per_core": cpu_per_core,
            "ram_total_mb": mem.total / (1024 * 1024),
            "ram_used_mb": mem.used / (1024 * 1024),
            "ram_pct": mem.percent,
            "proc_ram_mb": proc_mem.rss / (1024 * 1024),
            "proc_threads": self._process.num_threads(),
            "cpu_temp_c": temp,
        }


# ═══════════════════════════════════════════════════════════════
# Pipeline Runner (headless, looping)
# ═══════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """Runs pipeline headless with video looping for N seconds."""

    def __init__(self, video_paths, duration_s, rpi_mode=False):
        self.video_paths = video_paths
        self.duration_s = duration_s
        self.rpi_mode = rpi_mode

        # Stats
        self.frame_count = 0
        self.processed_count = 0
        self.yolo_calls = 0
        self.ae_calls = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.fps_history = []
        self.latency_history = []
        self._lock = threading.Lock()

    def run(self):
        cfg = Config.rpi_preset() if self.rpi_mode else Config()
        cfg.show_preview = False
        cfg.save_video = False
        cfg.ae_enabled = True

        if not self.rpi_mode:
            cfg.proc_width = 640
            cfg.proc_height = 480
            cfg.roi_top = 0.5
            cfg.ae_input_size = 64
            cfg.ae_threshold = 0.06
            cfg.ae_smooth_window = 5
            cfg.ae_smooth_min_hits = 2
            cfg.ae_recheck_interval = 15
            cfg.tracker_iou = 0.25
            cfg.tracker_max_lost = 15

        # Init
        from classifier_ncnn import YOLOClassifier
        cfg.sources = self.video_paths
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

        yolo_cooldown = 0
        ae_run_counter = 0
        ae_every_n = 3
        video_idx = 0
        loop_count = 0

        t_start = time.time()

        # Open first video
        cap = cv2.VideoCapture(self.video_paths[video_idx])
        print(f"  [Video] {os.path.basename(self.video_paths[video_idx])}")

        try:
            while True:
                elapsed_total = time.time() - t_start
                if elapsed_total >= self.duration_s:
                    break

                ret, raw_frame = cap.read()
                if not ret:
                    # Loop: next video or restart
                    video_idx = (video_idx + 1) % len(self.video_paths)
                    cap.release()
                    cap = cv2.VideoCapture(self.video_paths[video_idx])
                    loop_count += 1
                    if loop_count % len(self.video_paths) == 0:
                        pass  # full loop complete
                    continue

                self.frame_count += 1
                if self.frame_count % 3 != 0:
                    continue

                t0 = time.time()
                frame = cv2.resize(raw_frame, (cfg.proc_width, cfg.proc_height))
                roi = frame[roi_y_start:roi_y_end]

                # Motion
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

                # Tracker check
                all_tracked = True
                if motion_regions:
                    for (mx, my, mw2, mh), _ in motion_regions:
                        if not any(_iou((mx, my, mw2, mh), tb) > 0.3
                                   for tb in tracker.active_bboxes):
                            all_tracked = False
                            break

                # AE
                ae_says_anomaly = False
                if ae and motion_regions:
                    ae_run_counter += 1
                    run_ae = (ae_run_counter % ae_every_n == 0) or \
                             (not all_tracked and ae_run_counter % 2 == 0)
                    if run_ae:
                        eff_thresh = cfg.ae_threshold * mw
                        ae_says_anomaly, _, _ = ae.is_anomaly_smoothed(roi, eff_thresh)
                        self.ae_calls += 1

                # YOLO
                yolo.fetch()  # consume results
                det_candidates = []
                submit = False
                if not all_tracked:
                    submit = ae_says_anomaly if ae else bool(motion_regions)
                if yolo_cooldown > 0:
                    yolo_cooldown -= 1
                    submit = False
                if submit:
                    yolo.submit(roi)
                    self.yolo_calls += 1
                    yolo_cooldown = 5

                tracker.update(det_candidates, self.frame_count)

                # Timing
                elapsed = time.time() - t0
                self.processed_count += 1
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(elapsed, 0.001))
                self.latency_ms = 0.9 * self.latency_ms + 0.1 * (elapsed * 1000)

                # Sample every 30 frames
                if self.processed_count % 30 == 0:
                    with self._lock:
                        self.fps_history.append(self.fps)
                        self.latency_history.append(self.latency_ms)

                # Progress
                if self.processed_count % 200 == 0:
                    pct = elapsed_total / self.duration_s * 100
                    print(f"  [{pct:5.1f}%] Frame {self.frame_count} | "
                          f"FPS: {self.fps:.1f} | Lat: {self.latency_ms:.0f}ms | "
                          f"YOLO: {self.yolo_calls} | AE: {self.ae_calls}")

        finally:
            yolo.stop_async()
            cap.release()

        self.total_time = time.time() - t_start


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pipeline Benchmark")
    parser.add_argument("--duration", type=int, default=600, help="Duration in seconds (default: 600 = 10 min)")
    parser.add_argument("--rpi", action="store_true", help="Use RPi preset")
    parser.add_argument("--videos", nargs="+", default=None, help="Video files (default: test_video/*.mp4)")
    args = parser.parse_args()

    videos = args.videos or sorted(glob.glob("test_video/*.mp4"))
    if not videos:
        print("ERROR: No video files found. Put .mp4 files in test_video/ or use --videos")
        sys.exit(1)

    duration = args.duration
    os.makedirs("output", exist_ok=True)

    print("╔══════════════════════════════════════╗")
    print("║     Pipeline Benchmark               ║")
    print("╚══════════════════════════════════════╝\n")
    print(f"  Duration:  {duration}s ({duration/60:.0f} min)")
    print(f"  Videos:    {len(videos)} files (looping)")
    for v in videos:
        print(f"             • {os.path.basename(v)}")
    print(f"  RPi mode:  {'YES' if args.rpi else 'NO'}")
    print(f"  Platform:  {sys.platform}")
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  RAM:       {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()

    # Start system monitor
    monitor = SystemMonitor(sample_interval=2.0)
    monitor.start()

    # Run pipeline
    runner = BenchmarkRunner(videos, duration, rpi_mode=args.rpi)
    print("  Starting pipeline...\n")
    t_start = time.time()

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n  [Ctrl+C] Stopping early...")
    finally:
        monitor.stop()

    # Normalize timestamps
    if monitor.samples:
        t0 = monitor.samples[0]["elapsed_s"]
        for s in monitor.samples:
            s["elapsed_s"] -= t0

    # ── Write CSV ──
    csv_path = os.path.join("output", "benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elapsed_s", "timestamp",
            "cpu_total_pct", "cpu_max_core_pct",
            "ram_used_mb", "ram_pct", "proc_ram_mb",
            "proc_threads", "cpu_temp_c",
        ])
        for s in monitor.samples:
            writer.writerow([
                f"{s['elapsed_s']:.1f}",
                s["timestamp"],
                f"{s['cpu_total_pct']:.1f}",
                f"{s['cpu_max_core_pct']:.1f}",
                f"{s['ram_used_mb']:.0f}",
                f"{s['ram_pct']:.1f}",
                f"{s['proc_ram_mb']:.1f}",
                s["proc_threads"],
                f"{s['cpu_temp_c']:.1f}" if s["cpu_temp_c"] else "",
            ])
    print(f"\n  CSV saved: {csv_path} ({len(monitor.samples)} samples)")

    # ── Summary ──
    samples = monitor.samples
    if not samples:
        print("  No samples collected!")
        return

    cpu_avg = sum(s["cpu_total_pct"] for s in samples) / len(samples)
    cpu_max = max(s["cpu_max_core_pct"] for s in samples)
    ram_avg = sum(s["proc_ram_mb"] for s in samples) / len(samples)
    ram_max = max(s["proc_ram_mb"] for s in samples)
    threads_avg = sum(s["proc_threads"] for s in samples) / len(samples)
    threads_max = max(s["proc_threads"] for s in samples)

    temps = [s["cpu_temp_c"] for s in samples if s["cpu_temp_c"] is not None]
    temp_avg = sum(temps) / len(temps) if temps else None
    temp_max = max(temps) if temps else None

    fps_avg = sum(runner.fps_history) / len(runner.fps_history) if runner.fps_history else runner.fps
    fps_min = min(runner.fps_history) if runner.fps_history else runner.fps
    fps_max = max(runner.fps_history) if runner.fps_history else runner.fps
    lat_avg = sum(runner.latency_history) / len(runner.latency_history) if runner.latency_history else runner.latency_ms
    lat_max = max(runner.latency_history) if runner.latency_history else runner.latency_ms

    summary = []
    summary.append("═" * 55)
    summary.append("  BENCHMARK RESULTS")
    summary.append("═" * 55)
    summary.append(f"  Duration:        {runner.total_time:.1f}s ({runner.total_time/60:.1f} min)")
    summary.append(f"  Total frames:    {runner.frame_count}")
    summary.append(f"  Processed:       {runner.processed_count}")
    summary.append(f"  YOLO calls:      {runner.yolo_calls}")
    summary.append(f"  AE calls:        {runner.ae_calls}")
    summary.append("")
    summary.append("  ── FPS & Latency ──")
    summary.append(f"  FPS avg:         {fps_avg:.1f}")
    summary.append(f"  FPS range:       {fps_min:.1f} - {fps_max:.1f}")
    summary.append(f"  Latency avg:     {lat_avg:.0f}ms")
    summary.append(f"  Latency max:     {lat_max:.0f}ms")
    summary.append("")
    summary.append("  ── CPU ──")
    summary.append(f"  CPU avg:         {cpu_avg:.1f}%")
    summary.append(f"  CPU max (core):  {cpu_max:.1f}%")
    summary.append(f"  Cores:           {psutil.cpu_count()}")
    if temp_avg is not None:
        summary.append(f"  Temp avg:        {temp_avg:.1f}°C")
        summary.append(f"  Temp max:        {temp_max:.1f}°C")
    else:
        summary.append(f"  Temp:            N/A (sensor not available)")
    summary.append("")
    summary.append("  ── Memory ──")
    summary.append(f"  Process RAM avg: {ram_avg:.1f} MB")
    summary.append(f"  Process RAM max: {ram_max:.1f} MB")
    summary.append(f"  System RAM:      {psutil.virtual_memory().percent:.1f}% used")
    summary.append("")
    summary.append("  ── Threads ──")
    summary.append(f"  Threads avg:     {threads_avg:.1f}")
    summary.append(f"  Threads max:     {threads_max}")
    summary.append("═" * 55)

    # Print
    print()
    for line in summary:
        print(line)
    print()

    # Save
    summary_path = os.path.join("output", "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Benchmark run: {datetime.now().isoformat()}\n")
        f.write(f"Platform: {sys.platform}\n")
        f.write(f"Videos: {', '.join(os.path.basename(v) for v in videos)}\n\n")
        for line in summary:
            f.write(line + "\n")
    print(f"  Summary saved: {summary_path}")
    print(f"  CSV saved:     {csv_path}")
    print()


if __name__ == "__main__":
    main()
