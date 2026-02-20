"""Main pipeline: ties all modules together."""

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


def draw_results(frame, motion_regions, anomaly_crops, mean_error, fps):
    display = frame.copy()

    # Motion regions (green)
    for (x, y, w, h), _ in motion_regions:
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Anomaly crops (red box + AE score + YOLO detections)
    for crop in anomaly_crops:
        x, y, w, h = crop["bbox"]
        # Color: cyan for unknown, class-color for known
        is_known = len(crop.get("yolo", [])) > 0
        box_color = (0, 0, 255) if is_known else (255, 255, 0)  # red=known, cyan=unknown
        cv2.rectangle(display, (x, y), (x+w, y+h), box_color, 2)
        label = f"AE:{crop['score']:.3f}"
        if not is_known:
            label += " [unknown]"
        cv2.putText(display, label, (x, max(y-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        for det in crop.get("yolo", []):
            bx, by, bw, bh = det["bbox"]
            bx2, by2 = x + bx, y + by
            color = YOLO_COLORS[det["class_id"] % len(YOLO_COLORS)]
            cv2.rectangle(display, (bx2, by2), (bx2+bw, by2+bh), color, 2)
            cv2.putText(display, f"{det['class_name']} {det['confidence']:.2f}",
                        (bx2, max(by2-4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Status bar
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, f"AE Error: {mean_error:.4f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2)
    cv2.putText(display, f"Motion: {len(motion_regions)}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, f"Anomalies: {len(anomaly_crops)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return display


# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════

class JSONLLogger:
    """Writes one JSON line per anomaly crop. Separates known/unknown."""

    def __init__(self, path):
        self._file = open(path, "w")
        self._prev = {}  # cam_id -> set of crop keys for dedup

    def log(self, frame_num, cam_id, mean_error, crop, kind="known"):
        yolo = crop.get("yolo", [])
        top_cls = yolo[0]["class_name"] if yolo else "unknown_anomaly"
        top_conf = yolo[0]["confidence"] if yolo else 0.0

        # Dedup: skip if same bbox+class as previous frame on same camera
        key = (crop["bbox"], top_cls)
        prev = self._prev.get(cam_id, set())
        if key in prev:
            return
        self._prev[cam_id] = {key}

        event = {
            "timestamp": datetime.now().isoformat(),
            "frame": frame_num,
            "camera": cam_id,
            "kind": kind,  # "known" or "unknown"
            "ae_error": round(mean_error, 4),
            "crop_score": round(crop["score"], 4),
            "bbox": {"x": crop["bbox"][0], "y": crop["bbox"][1],
                     "w": crop["bbox"][2], "h": crop["bbox"][3]},
            "class": top_cls,
            "yolo_confidence": top_conf,
            "yolo_detections": [
                {"class": d["class_name"], "confidence": d["confidence"],
                 "bbox": {"x": d["bbox"][0], "y": d["bbox"][1],
                          "w": d["bbox"][2], "h": d["bbox"][3]}}
                for d in yolo
            ],
        }
        self._file.write(json.dumps(event) + "\n")
        return event  # return for image saving

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
    print("╚══════════════════════════════════════╝\n")

    os.makedirs(cfg.output_dir, exist_ok=True)
    known_dir = os.path.join(cfg.output_dir, "known")
    unknown_dir = os.path.join(cfg.output_dir, "unknown")
    os.makedirs(known_dir, exist_ok=True)
    os.makedirs(unknown_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "detections.jsonl")

    camera = MultiCamera(cfg)
    src_fps = camera.src_fps or 30.0
    motion_dets = [MotionDetector(cfg) for _ in cfg.sources]
    ae = AnomalyDetector(cfg)
    cropper = Cropper(cfg)
    validator = Validator(cfg)
    yolo = load_classifier(cfg)
    logger = JSONLLogger(log_path)
    vid_writer = VideoWriterWrapper(cfg.output_dir, cfg.proc_width, cfg.proc_height)

    print(f"[Pipeline] Log → {log_path}")
    print(f"[Pipeline] Known anomalies → {known_dir}/")
    print(f"[Pipeline] Unknown anomalies → {unknown_dir}/")
    print("[Pipeline] Running... Press 'q' to quit.\n")

    frame_count = 0
    skip_n = cfg.skip_frames
    motion_total = 0
    anomaly_total = 0
    known_total = 0
    unknown_total = 0
    img_counter = 0
    t_start = time.time()
    fps = 0.0

    if skip_n > 0:
        print(f"[Pipeline] Frame skip: processing every {skip_n} frame(s)")
    if cfg.max_crops < 999:
        print(f"[Pipeline] Max crops per frame: {cfg.max_crops}")

    try:
        while True:
            frames = camera.read_all()
            if all(f is None for f in frames):
                print("[Pipeline] All sources exhausted.")
                break

            frame_count += 1

            # Frame skipping — still read (to advance video) but skip heavy ops
            if skip_n > 0 and (frame_count % skip_n != 0):
                # Feed frame to motion detector to keep BG model updated
                for i, frame in enumerate(frames):
                    if frame is not None:
                        motion_dets[i].detect(frame)
                continue

            for i, frame in enumerate(frames):
                if frame is None:
                    continue

                t_frame = time.time()
                mean_error = 0.0
                anomaly_crops = []

                # Stage 1: Motion detection → binary mask
                mask, motion_regions = motion_dets[i].detect(frame)

                if motion_regions:
                    motion_total += 1

                    # Stage 2: Mask moving objects (black out) before AE
                    #   Prevents shadows/people from triggering false anomalies
                    masked_frame = frame.copy()
                    mask_inv = (mask > 0)  # True where motion exists
                    masked_frame[mask_inv] = 0  # black out motion pixels

                    # Stage 3: Autoencoder on masked frame
                    error_map, mean_error = ae.infer(masked_frame)

                    # Zero out error in masked regions (black boxes aren't anomalies)
                    error_map[mask_inv] = 0.0
                    # Recompute mean error excluding masked pixels
                    unmasked = ~mask_inv
                    if unmasked.any():
                        mean_error = float(error_map[unmasked].mean())
                    else:
                        mean_error = 0.0

                    if mean_error > cfg.ae_threshold:
                        # Stage 4: Crop from ORIGINAL frame using cleaned error map
                        crops = cropper.extract(frame, error_map, cfg.ae_threshold)

                        # Stage 5: Validate + YOLO classify
                        yolo_count = 0
                        for crop in crops:
                            if yolo_count >= cfg.max_crops:
                                break
                            if validator.is_valid(crop["image"]):
                                crop["yolo"] = yolo.infer(crop["image"])
                                anomaly_crops.append(crop)
                                yolo_count += 1

                        # Stage 6: Separate known / unknown + log + save images
                        if anomaly_crops:
                            anomaly_total += 1
                            for crop in anomaly_crops:
                                has_det = len(crop.get("yolo", [])) > 0
                                kind = "known" if has_det else "unknown"
                                event = logger.log(frame_count, i, mean_error, crop, kind)
                                if event is not None:  # not deduped
                                    img_counter += 1
                                    if has_det:
                                        known_total += 1
                                        cls = crop["yolo"][0]["class_name"]
                                        fname = f"{img_counter:04d}_f{frame_count}_{cls}.jpg"
                                        cv2.imwrite(os.path.join(known_dir, fname), crop["image"])
                                    else:
                                        unknown_total += 1
                                        fname = f"{img_counter:04d}_f{frame_count}_unknown.jpg"
                                        cv2.imwrite(os.path.join(unknown_dir, fname), crop["image"])

                # Draw & write video
                elapsed = time.time() - t_frame
                fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))
                display = draw_results(frame, motion_regions, anomaly_crops, mean_error, fps)
                vid_writer.write(i, display)

                if cfg.show_preview:
                    cv2.imshow(f"Camera {i}", display)
                    if mean_error > 0:
                        err_vis = cv2.resize(error_map, (480, 270))
                        err_vis = (err_vis * 255 / max(err_vis.max(), 0.001)).astype(np.uint8)
                        err_vis = cv2.applyColorMap(err_vis, cv2.COLORMAP_JET)
                        cv2.imshow(f"Error Map {i}", err_vis)

            if frame_count % 30 == 0:
                print(f"  Frame {frame_count} | FPS: {fps:.1f} | "
                      f"Motion: {motion_total} | Anomalies: {anomaly_total}")

            if cfg.show_preview:
                if cfg.realtime:
                    # Throttle to real video speed
                    proc_ms = int((time.time() - t_frame) * 1000)
                    wait_ms = max(1, int(1000 / src_fps) - proc_ms)
                else:
                    wait_ms = 1  # process as fast as possible
                key = cv2.waitKey(wait_ms)
                if key == ord("q") or key == 27:
                    print("\n[Pipeline] Quit requested")
                    break

    finally:
        camera.release()
        vid_writer.release()
        logger.close()
        cv2.destroyAllWindows()

    elapsed_total = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"  Pipeline Complete!")
    print(f"{'='*50}")
    print(f"  Total frames:   {frame_count}")
    print(f"  Motion frames:  {motion_total}")
    print(f"  Anomaly frames: {anomaly_total}")
    print(f"  Known anomalies:  {known_total}")
    print(f"  Unknown anomalies: {unknown_total}")
    print(f"  Avg FPS:        {frame_count / max(elapsed_total, 0.001):.1f}")
    print(f"  Time:           {elapsed_total:.1f}s")
    print(f"{'='*50}")


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
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--skip-frames", type=int, default=None,
                        help="Process every Nth frame (0=all)")
    parser.add_argument("--max-crops", type=int, default=None,
                        help="Max anomaly crops sent to YOLO per frame")
    parser.add_argument("--realtime", action="store_true",
                        help="Throttle display to source video FPS")
    parser.add_argument("--proc-width", type=int, default=None)
    parser.add_argument("--proc-height", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
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

    run(cfg)


if __name__ == "__main__":
    main()
