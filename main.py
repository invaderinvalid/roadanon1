"""Road Anomaly Detection — Tkinter GUI.

Two modes:
  1. Pi Camera / USB webcam (live)
  2. Video file playback

Shows live video with detections, AE/YOLO status, and anomaly alerts.
Uses pipeline modules directly — no pipeline.py modification.

Requirements:
  brew install python-tk@3.14   (for macOS homebrew Python)
  pip install Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import time
import os
import threading
from datetime import datetime
from PIL import Image, ImageTk
from collections import deque

from config import Config
from motion_detect import MotionDetector
from tracker import SimpleTracker, _iou


# ═══════════════════════════════════════════════════════════════
# Detection Engine (runs in background thread)
# ═══════════════════════════════════════════════════════════════

CLASS_COLORS_BGR = {
    "road_damage": (0, 0, 255),
    "speed_bump": (0, 165, 255),
    "unsurfaced_road": (220, 220, 0),
}


class DetectionEngine:
    """Runs detection using pipeline modules in a background thread."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._frame_queue = deque(maxlen=2)
        self._stats = {}
        self._alerts = deque(maxlen=50)
        self._lock = threading.Lock()

    @property
    def running(self):
        return self._running

    def get_frame(self):
        with self._lock:
            return self._frame_queue.popleft() if self._frame_queue else None

    def get_stats(self):
        with self._lock:
            return self._stats.copy()

    def get_new_alerts(self):
        with self._lock:
            alerts = list(self._alerts)
            self._alerts.clear()
            return alerts

    def start(self, source, mode="video"):
        if self._running:
            self.stop()
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(source, mode), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def _run(self, source, mode):
        cfg = Config()
        cfg.show_preview = False
        cfg.save_video = False
        cfg.roi_top = 0.5
        cfg.ae_enabled = True
        cfg.ae_input_size = 64
        cfg.ae_threshold = 0.06
        cfg.ae_smooth_window = 5
        cfg.ae_smooth_min_hits = 2
        cfg.ae_recheck_interval = 15
        cfg.tracker_iou = 0.25
        cfg.tracker_max_lost = 15
        cfg.yolo_conf = 0.25

        if mode == "camera":
            cfg.proc_width = 320
            cfg.proc_height = 240
            source = int(source) if str(source).isdigit() else 0
        else:
            cfg.proc_width = 640
            cfg.proc_height = 480

        cfg.sources = [source]

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            with self._lock:
                self._alerts.append(("ERROR", "Cannot open source", "#ff0000"))
            self._running = False
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode != "camera" else 0

        # Init modules
        from classifier_ncnn import YOLOClassifier
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

        frame_count = 0
        yolo_calls = 0
        fps = 0.0
        load_avg_ms = 0.0
        last_ae_error = None
        yolo_cooldown = 0
        ae_says_anomaly = False
        motion_pct = 0.0

        try:
            while self._running:
                ret, raw_frame = cap.read()
                if not ret:
                    with self._lock:
                        self._alerts.append(("INFO", "Video ended", "#888888"))
                    break

                frame_count += 1
                t0 = time.time()

                frame = cv2.resize(raw_frame, (cfg.proc_width, cfg.proc_height))
                roi = frame[roi_y_start:roi_y_end]

                # Motion
                mask, motion_regions = motion_det.detect(roi)
                motion_pct = 0.0
                if motion_regions:
                    total_m = sum(a for _, a in motion_regions)
                    motion_pct = (total_m / roi_area) * 100.0

                # Motion weight
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
                    run_ae = not all_tracked or (frame_count % cfg.ae_recheck_interval == 0)
                    if run_ae:
                        eff_thresh = cfg.ae_threshold * mw
                        ae_says_anomaly, ae_err, _ = ae.is_anomaly_smoothed(roi, eff_thresh)
                        last_ae_error = ae_err

                # YOLO fetch
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

                # YOLO submit
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

                # Tracker
                new_tracks, active_tracks, gone_tracks = tracker.update(
                    det_candidates, frame_count)
                for t in new_tracks:
                    for dc in det_candidates:
                        if dc["bbox"] == t.bbox:
                            t.label = dc["label"]
                            t.confidence = dc["confidence"]
                            break

                # Timing
                elapsed = time.time() - t0
                fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))
                load_avg_ms = 0.9 * load_avg_ms + 0.1 * (elapsed * 1000)

                # Draw on frame
                display = frame.copy()
                # ROI line
                cv2.line(display, (0, roi_y_start), (cfg.proc_width, roi_y_start),
                         (0, 255, 255), 1)
                # Motion boxes
                if motion_regions:
                    for (mx, my, mw2, mh), _ in motion_regions:
                        cv2.rectangle(display,
                                      (mx, roi_y_start + my),
                                      (mx + mw2, roi_y_start + my + mh),
                                      (0, 180, 0), 1)
                # Track boxes
                for t in active_tracks:
                    tx, ty, tw, th = t.bbox
                    fy = roi_y_start + ty
                    color = CLASS_COLORS_BGR.get(t.label, (0, 255, 255))
                    cv2.rectangle(display, (tx, fy), (tx + tw, fy + th), color, 2)
                    if t.label:
                        lbl = f"{t.label} {t.confidence:.0%}"
                        cv2.putText(display, lbl, (tx, max(fy - 6, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                                    cv2.LINE_AA)

                # Convert BGR → RGB
                frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame_queue.append(frame_rgb)

                # Alerts for new tracks
                for t in new_tracks:
                    if t.label:
                        with self._lock:
                            self._alerts.append((
                                t.label,
                                f"{t.label} ({t.confidence:.0%}) — Track #{t.id}",
                                {"road_damage": "#ff3c3c", "speed_bump": "#ffa500",
                                 "unsurfaced_road": "#00dcdc"}.get(t.label, "#ffff00")
                            ))

                # Stats
                with self._lock:
                    self._stats = {
                        "fps": fps,
                        "frame": frame_count,
                        "total_frames": total_frames,
                        "motion_pct": motion_pct,
                        "ae_error": last_ae_error,
                        "ae_threshold": cfg.ae_threshold * mw if ae else None,
                        "yolo_busy": yolo.is_busy,
                        "yolo_calls": yolo_calls,
                        "yolo_cooldown": yolo_cooldown,
                        "tracks": len(active_tracks),
                        "latency_ms": load_avg_ms,
                        "anomaly": ae_says_anomaly,
                        "ae_on": ae is not None,
                    }

                # Throttle for video files
                target_delay = 1.0 / src_fps - elapsed
                if target_delay > 0 and isinstance(source, str):
                    time.sleep(target_delay * 0.8)

        finally:
            yolo.stop_async()
            cap.release()
            self._running = False


# ═══════════════════════════════════════════════════════════════
# Tkinter GUI
# ═══════════════════════════════════════════════════════════════

BG_DARK = "#0f0f1a"
BG_PANEL = "#16213e"
BG_ENTRY = "#0f3460"
FG_TEXT = "#e0e0e0"
FG_ACCENT = "#00d4ff"
FG_DIM = "#666666"
GREEN = "#00e676"
YELLOW = "#ffd600"
RED = "#ff1744"
ORANGE = "#ff9100"


class AnomalyDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Road Anomaly Detection")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1020x680")
        self.root.minsize(850, 550)

        self.engine = DetectionEngine()
        self._photo = None
        self._build_ui()
        self._poll()

    # ── Build UI ──────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG_DARK)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Label(hdr, text="🛣  Road Anomaly Detection", bg=BG_DARK, fg=FG_ACCENT,
                 font=("Helvetica", 17, "bold")).pack(side="left")
        self.status_dot = tk.Label(hdr, text="● STOPPED", bg=BG_DARK, fg=FG_DIM,
                                    font=("Helvetica", 12))
        self.status_dot.pack(side="right")

        # Main split
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        # LEFT — video
        left = tk.Frame(body, bg="#000000", bd=2, relief="groove")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.video_lbl = tk.Label(left, bg="#000000",
                                   text="Select a mode and press Start",
                                   fg="#444", font=("Helvetica", 15))
        self.video_lbl.pack(fill="both", expand=True)

        # RIGHT — sidebar
        right = tk.Frame(body, bg=BG_DARK, width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # ── Mode selector ──
        mode_box = tk.LabelFrame(right, text=" Mode ", bg=BG_PANEL, fg=FG_ACCENT,
                                  font=("Helvetica", 10, "bold"), bd=1, padx=8, pady=6)
        mode_box.pack(fill="x", pady=(0, 6))

        self.mode_var = tk.StringVar(value="video")
        for text, val in [("📹  Pi Camera / Webcam", "camera"),
                          ("🎬  Video File", "video")]:
            tk.Radiobutton(mode_box, text=text, variable=self.mode_var, value=val,
                           bg=BG_PANEL, fg=FG_TEXT, selectcolor=BG_ENTRY,
                           activebackground=BG_PANEL, activeforeground=FG_ACCENT,
                           font=("Helvetica", 10), anchor="w"
                           ).pack(fill="x", pady=1)

        # File chooser
        ff = tk.Frame(mode_box, bg=BG_PANEL)
        ff.pack(fill="x", pady=(4, 2))
        self.file_var = tk.StringVar()
        tk.Entry(ff, textvariable=self.file_var, bg=BG_ENTRY, fg=FG_TEXT,
                 insertbackground="#fff", font=("Consolas", 9), bd=0
                 ).pack(side="left", fill="x", expand=True, ipady=3)
        tk.Button(ff, text="…", command=self._browse, bg=BG_ENTRY, fg=FG_TEXT,
                  font=("Helvetica", 10), bd=0, padx=8).pack(side="right", padx=(4, 0))

        # Camera ID
        cf = tk.Frame(mode_box, bg=BG_PANEL)
        cf.pack(fill="x", pady=(2, 0))
        tk.Label(cf, text="Cam ID:", bg=BG_PANEL, fg=FG_DIM,
                 font=("Helvetica", 9)).pack(side="left")
        self.cam_var = tk.StringVar(value="0")
        tk.Entry(cf, textvariable=self.cam_var, width=4, bg=BG_ENTRY, fg=FG_TEXT,
                 font=("Consolas", 10), bd=0).pack(side="left", padx=4, ipady=2)

        # ── Controls ──
        ctrl = tk.Frame(right, bg=BG_DARK)
        ctrl.pack(fill="x", pady=(0, 6))
        self.btn_start = tk.Button(ctrl, text="▶  Start", command=self._start,
                                    bg="#00a86b", fg="white",
                                    font=("Helvetica", 11, "bold"), bd=0, pady=5)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.btn_stop = tk.Button(ctrl, text="■  Stop", command=self._stop,
                                   bg="#c0392b", fg="white",
                                   font=("Helvetica", 11, "bold"), bd=0, pady=5,
                                   state="disabled")
        self.btn_stop.pack(side="right", fill="x", expand=True)

        # ── Stats panel ──
        stats_box = tk.LabelFrame(right, text=" Pipeline Stats ", bg=BG_PANEL, fg=FG_ACCENT,
                                    font=("Helvetica", 10, "bold"), bd=1, padx=8, pady=4)
        stats_box.pack(fill="x", pady=(0, 6))

        self.stat_lbls = {}
        for label, key in [("FPS", "fps"), ("Frame", "frame"), ("Latency", "latency"),
                           ("Motion %", "motion"), ("AE Error", "ae_error"),
                           ("AE Thresh", "ae_thresh"), ("Anomaly", "anomaly"),
                           ("YOLO", "yolo"), ("Tracks", "tracks")]:
            row = tk.Frame(stats_box, bg=BG_PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=BG_PANEL, fg=FG_DIM,
                     font=("Consolas", 9), width=10, anchor="w").pack(side="left")
            v = tk.Label(row, text="—", bg=BG_PANEL, fg=FG_TEXT,
                         font=("Consolas", 10, "bold"), anchor="w")
            v.pack(side="left", fill="x")
            self.stat_lbls[key] = v

        # AE bar canvas
        self.ae_canvas = tk.Canvas(stats_box, bg=BG_PANEL, height=12, bd=0,
                                    highlightthickness=0)
        self.ae_canvas.pack(fill="x", pady=(2, 4))

        # ── Anomaly alerts ──
        alert_box = tk.LabelFrame(right, text=" ⚠ Anomaly Alerts ", bg=BG_PANEL,
                                    fg="#ff6b6b", font=("Helvetica", 10, "bold"),
                                    bd=1, padx=4, pady=4)
        alert_box.pack(fill="both", expand=True)

        self.alert_text = tk.Text(alert_box, bg="#080818", fg=RED,
                                   font=("Consolas", 9), wrap="word", bd=0,
                                   state="disabled", height=6)
        self.alert_text.pack(fill="both", expand=True)
        self.alert_text.tag_configure("road_damage", foreground="#ff3c3c")
        self.alert_text.tag_configure("speed_bump", foreground="#ffa500")
        self.alert_text.tag_configure("unsurfaced_road", foreground="#00dcdc")
        self.alert_text.tag_configure("info", foreground="#888888")
        self.alert_text.tag_configure("ts", foreground="#444444")

    # ── Actions ───────────────────────────────────────────────

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("All", "*.*")])
        if p:
            self.file_var.set(p)

    def _start(self):
        mode = self.mode_var.get()
        if mode == "video":
            src = self.file_var.get()
            if not src or not os.path.exists(src):
                messagebox.showerror("Error", "Select a valid video file.")
                return
        else:
            src = self.cam_var.get()

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_dot.config(text="● RUNNING", fg=GREEN)
        self._add_alert("info", f"Started — {mode}: {os.path.basename(str(src))}")
        self.engine.start(src, mode)

    def _stop(self):
        self.engine.stop()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_dot.config(text="● STOPPED", fg=FG_DIM)
        self._add_alert("info", "Stopped")

    def _add_alert(self, tag, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.alert_text.config(state="normal")
        self.alert_text.insert("end", f"[{ts}] ", "ts")
        self.alert_text.insert("end", f"⚠ {text}\n", tag)
        self.alert_text.see("end")
        self.alert_text.config(state="disabled")

    # ── Polling loop ──────────────────────────────────────────

    def _poll(self):
        """Update GUI from engine state — runs on main thread every 33ms."""

        # Frame
        frame = self.engine.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            vw = max(self.video_lbl.winfo_width(), 100)
            vh = max(self.video_lbl.winfo_height(), 100)
            scale = min(vw / w, vh / h, 2.0)
            if abs(scale - 1.0) > 0.05:
                nw, nh = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (nw, nh))
            img = Image.fromarray(frame)
            self._photo = ImageTk.PhotoImage(img)
            self.video_lbl.config(image=self._photo, text="")

        # Alerts
        for tag, text, color in self.engine.get_new_alerts():
            self._add_alert(tag, text)

        # Stats
        s = self.engine.get_stats()
        if s:
            self._set_stat("fps", f"{s.get('fps', 0):.1f}")
            f = s.get("frame", 0)
            tf = s.get("total_frames", 0)
            self._set_stat("frame", f"{f}/{tf}" if tf else str(f))

            lat = s.get("latency_ms", 0)
            self._set_stat("latency", f"{lat:.0f} ms",
                           GREEN if lat < 33 else YELLOW if lat < 66 else RED)

            self._set_stat("motion", f"{s.get('motion_pct', 0):.1f}%")

            ae_err = s.get("ae_error")
            ae_thresh = s.get("ae_threshold")
            if ae_err is not None:
                c = RED if s.get("anomaly") else GREEN
                self._set_stat("ae_error", f"{ae_err:.4f}", c)
                self._draw_ae_bar(ae_err, ae_thresh or 0.06)
            else:
                self._set_stat("ae_error", "off" if not s.get("ae_on") else "—")
            self._set_stat("ae_thresh",
                           f"{ae_thresh:.4f}" if ae_thresh else "—")

            anom = s.get("anomaly", False)
            self._set_stat("anomaly", "⚠ YES" if anom else "no",
                           RED if anom else GREEN)

            ybusy = s.get("yolo_busy")
            ycd = s.get("yolo_cooldown", 0)
            yn = s.get("yolo_calls", 0)
            if ybusy:
                self._set_stat("yolo", f"BUSY ({yn})", YELLOW)
            elif ycd > 0:
                self._set_stat("yolo", f"CD:{ycd} ({yn})", FG_ACCENT)
            else:
                self._set_stat("yolo", f"READY ({yn})", GREEN)

            self._set_stat("tracks", str(s.get("tracks", 0)))

        # Auto-stop
        if not self.engine.running and self.btn_stop["state"] == "normal":
            self._stop()

        self.root.after(33, self._poll)

    def _set_stat(self, key, value, color=None):
        if key in self.stat_lbls:
            self.stat_lbls[key].config(text=value, fg=color or FG_TEXT)

    def _draw_ae_bar(self, error, threshold):
        c = self.ae_canvas
        c.delete("all")
        w = c.winfo_width()
        if w < 10:
            return
        max_val = max(threshold * 3, 0.15)
        err_px = int(min(error / max_val, 1.0) * w)
        thr_px = int(min(threshold / max_val, 1.0) * w)
        # Background
        c.create_rectangle(0, 0, w, 12, fill="#222", outline="")
        # Error fill
        fill_color = "#ff1744" if error > threshold else "#00e676"
        c.create_rectangle(0, 0, err_px, 12, fill=fill_color, outline="")
        # Threshold line
        c.create_line(thr_px, 0, thr_px, 12, fill="white", width=2)

    # ── Lifecycle ─────────────────────────────────────────────

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.engine.stop()
        self.root.destroy()


if __name__ == "__main__":
    app = AnomalyDetectorGUI()
    app.run()
