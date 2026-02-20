"""Lightweight IoU tracker for deduplicating anomalies across frames.

Tracks anomaly crops so the same pothole/bump isn't logged 100+ times
as it moves through the frame.  No Kalman, no deep features — just
greedy IoU matching on bounding boxes.  Very cheap for RPi.

Lifecycle:
  NEW       →  object first seen, trigger log + YOLO + save image
  ACTIVE    →  object still visible, skip YOLO (carry forward label)
  LOST(n)   →  not matched for n frames
  GONE      →  lost > max_lost, removed (optionally trigger exit-log)
"""


def _iou(a, b):
    """Compute IoU between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    __slots__ = ("id", "bbox", "first_frame", "last_frame",
                 "hits", "lost", "score", "label", "confidence",
                 "yolo_dets", "logged", "image")

    def __init__(self, tid, bbox, frame_num, score=0.0):
        self.id = tid
        self.bbox = bbox          # (x, y, w, h) in frame coords
        self.first_frame = frame_num
        self.last_frame = frame_num
        self.hits = 1
        self.lost = 0
        self.score = score        # AE error score
        self.label = None         # class name from YOLO (or None)
        self.confidence = 0.0     # YOLO confidence
        self.yolo_dets = []       # full YOLO detections list
        self.logged = False
        self.image = None         # crop image (saved on first detection)


class SimpleTracker:
    """Greedy IoU tracker.  O(T*D) per frame — fine for <20 tracks."""

    def __init__(self, iou_thresh=0.3, max_lost=8, min_hits=1):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost    # frames before removing a track
        self.min_hits = min_hits    # hits required before a track is "confirmed"
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections, frame_num):
        """
        Args:
            detections: list of dicts with keys:
                bbox: (x, y, w, h)
                score: float (AE error)
                image: np.ndarray (optional, crop)
            frame_num: int

        Returns:
            new_tracks:    list[Track] — just created this frame (run YOLO on these)
            active_tracks: list[Track] — all currently alive tracks
            gone_tracks:   list[Track] — removed this frame
        """
        new_tracks = []
        gone_tracks = []

        if not detections:
            # Nothing detected — age all tracks
            alive = []
            for t in self.tracks:
                t.lost += 1
                if t.lost > self.max_lost:
                    gone_tracks.append(t)
                else:
                    alive.append(t)
            self.tracks = alive
            return new_tracks, list(self.tracks), gone_tracks

        det_bboxes = [d["bbox"] for d in detections]
        det_scores = [d.get("score", 0.0) for d in detections]
        det_images = [d.get("image") for d in detections]

        # ── Greedy IoU matching ──
        matched_det = set()
        matched_trk = set()

        if self.tracks:
            pairs = []
            for ti, t in enumerate(self.tracks):
                for di, db in enumerate(det_bboxes):
                    iou = _iou(t.bbox, db)
                    if iou >= self.iou_thresh:
                        pairs.append((iou, ti, di))
            pairs.sort(key=lambda x: x[0], reverse=True)

            for iou_val, ti, di in pairs:
                if ti in matched_trk or di in matched_det:
                    continue
                matched_trk.add(ti)
                matched_det.add(di)
                t = self.tracks[ti]
                t.bbox = det_bboxes[di]
                t.last_frame = frame_num
                t.hits += 1
                t.lost = 0
                t.score = det_scores[di]

        # ── Create new tracks for unmatched detections ──
        for di in range(len(detections)):
            if di not in matched_det:
                t = Track(self._next_id, det_bboxes[di], frame_num, det_scores[di])
                t.image = det_images[di]
                self._next_id += 1
                self.tracks.append(t)
                new_tracks.append(t)

        # ── Age unmatched tracks ──
        alive = []
        for ti, t in enumerate(self.tracks):
            if ti not in matched_trk and t not in new_tracks:
                t.lost += 1
            if t.lost > self.max_lost:
                gone_tracks.append(t)
            else:
                alive.append(t)
        self.tracks = alive

        return new_tracks, list(self.tracks), gone_tracks

    @property
    def active_bboxes(self):
        """Return bboxes of active (not lost) tracks for suppression."""
        return [t.bbox for t in self.tracks if t.lost == 0]
