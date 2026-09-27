"""
Vehicle detection presets, shared by the app (transtar_feed.py) and the eval
harness (eval/evaluate.py), so the scores in eval/ are for exactly what runs.

Measured on eval/ (12 hand-labeled daytime frames, 287 vehicles, each preset
at its confidence cutoff; timings on an Apple-silicon GPU):

    preset        recall  precision  ms/frame
    ensemble        89%       94%       ~820   YOLO11x @1280 + RT-DETR-x, boxes fused (most accurate)
    yolo11s-hd      83%       84%        ~80   YOLO11s @1280 (best fast option)

The setups these replaced, YOLO11m and YOLOv8s at 640 px with confidence
>= 0.25, found only 46% and 49% of the cars (precision 99% and 98%): the 0.25
cutoff threw away real detections, and 320-427 px wide frames need upscaling
(imgsz 1280) for small cars to register. Going past 1280 made things worse
(too blurry).
"""

import threading
from pathlib import Path

import numpy as np
import torch
from ultralytics import RTDETR, YOLO

HERE = Path(__file__).parent
WEIGHTS = {
    "yolo11s": HERE / "weights" / "yolo11s.pt",
    "yolo11x": HERE / "weights" / "yolo11x.pt",
    "rtdetr-x": HERE / "weights" / "rtdetr-x.pt",
}
# COCO class id -> name, for the classes the article counts as vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# GPU when there is one (NVIDIA, or Apple silicon: ~3x faster than its CPU with identical boxes), else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# PyTorch's MPS backend crashes (Metal command-buffer assertion) when two threads
# run models at once, so every prediction takes this lock. The GPU would run
# them one at a time anyway.
GPU_LOCK = threading.Lock()

PRESETS = {
    "ensemble": dict(name="Ensemble", members=[("yolo11x", 1280), ("rtdetr-x", 640)], conf=0.15,
                     detail="YOLO11x (1280 px input) and RT-DETR-x, boxes fused where they agree; confidence >= 0.15"),
    "yolo11s-hd": dict(name="YOLO11s HD", members=[("yolo11s", 1280)], conf=0.10,
                       detail="YOLO11s with frames upscaled to 1280 px; confidence >= 0.10"),
}


def load(model_name):
    path = WEIGHTS[model_name]
    path.parent.mkdir(exist_ok=True)  # Ultralytics downloads missing weights to this path
    return RTDETR(str(path)) if model_name.startswith("rtdetr") else YOLO(str(path))


def predict(model, frame, imgsz, conf):
    """(N, 6) array of x1 y1 x2 y2 score class_id for vehicle classes.
    agnostic_nms: otherwise one vehicle can get both a car and a truck box."""
    with GPU_LOCK:
        r = model.predict(frame, conf=conf, imgsz=imgsz, classes=list(VEHICLE_CLASSES), agnostic_nms=True,
                          device=DEVICE, verbose=False)[0]
    b = r.boxes
    return np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()].reshape(-1, 6)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def fuse(box_sets, iou_thr=0.55):
    """Weighted box fusion. box_sets: one (N, 5 or 6) array per model
    (x1 y1 x2 y2 score [class]). Overlapping boxes from different models are
    merged into one box (coordinates averaged, weighted by score), scored by
    the mean over ALL models of each model's best score in the cluster (a
    model that missed the vehicle counts as 0), so agreement raises confidence.
    The class comes from the highest-scoring member. Returns (N, 6)."""
    rows = [np.c_[b[:, :5], b[:, 5] if b.shape[1] > 5 else np.zeros(len(b)), np.full(len(b), k)]
            for k, b in enumerate(box_sets) if len(b)]
    allb = np.concatenate(rows) if rows else np.zeros((0, 7))
    allb = allb[np.argsort(-allb[:, 4])]
    clusters = []  # [fused xyxy, members]
    for b in allb:
        if clusters:
            ov = iou(b[None, :4], np.array([c[0] for c in clusters]))[0]
            j = int(np.argmax(ov))
            if ov[j] >= iou_thr:
                clusters[j][1].append(b)
                m = np.array(clusters[j][1])
                clusters[j][0] = (m[:, :4] * m[:, 4:5]).sum(0) / m[:, 4].sum()
                continue
        clusters.append([b[:4].copy(), [b]])
    out = []
    for fused, members in clusters:
        best = {}
        for m in members:
            best[m[6]] = max(best.get(m[6], 0.0), m[4])
        out.append(np.r_[fused, sum(best.values()) / len(box_sets), members[0][5]])
    return np.array(out).reshape(-1, 6)


class PresetDetector:
    """Runs one preset. detect(frame) -> (boxes, class names, confidences).
    Not thread-safe: give each thread its own instance."""

    def __init__(self, preset):
        self.id = preset
        cfg = PRESETS[preset]
        self.name, self.detail, self.conf = cfg["name"], cfg["detail"], cfg["conf"]
        self.members = [(load(m), imgsz) for m, imgsz in cfg["members"]]

    def raw(self, frame):
        """(N, 6) boxes after fusion, before the confidence cutoff."""
        if len(self.members) == 1:
            model, imgsz = self.members[0]
            return predict(model, frame, imgsz, 0.05)
        return fuse([predict(model, frame, imgsz, 0.05) for model, imgsz in self.members])

    def detect(self, frame):
        d = self.raw(frame)
        d = d[d[:, 4] >= self.conf]
        return d[:, :4].tolist(), [VEHICLE_CLASSES[int(c)] for c in d[:, 5]], d[:, 4].tolist()
