"""
Rough vehicle speeds from two frames of the same camera, with no calibration.

1. Motion: pyramidal Lucas-Kanade optical flow on corner points inside each
   detection box, kept only if they track back to where they started
   (forward-backward check); the median point motion is the vehicle's motion.
2. Scale: the vehicle is its own ruler. Its box width is its real width when
   it drives toward/away from the camera and its real length when it crosses
   the view. Along the direction it moves, its image extent is its length
   (foreshortened exactly like its motion, so perspective cancels), once the
   part of the box that's just the vehicle's height is subtracted.

This assumes typical vehicle dimensions and a flat road, so treat results as
+-30% or worse, especially for small, distant vehicles.
"""

import math

import cv2
import numpy as np

# Typical (width, length, height) in meters
DIMENSIONS = {"car": (1.8, 4.5, 1.5), "truck": (2.0, 5.8, 2.0), "bus": (2.6, 12.0, 3.2), "motorcycle": (0.8, 2.2, 1.4)}
MIN_BOX_PX = 12        # smaller boxes are too coarse to measure
STOPPED_PX = 0.5       # median motion below this is within measurement noise
MAX_MPH = 110          # anything faster is a tracking error
MPS_TO_MPH = 2.23694


def box_motions(gray_a, gray_b, boxes, max_points=15):
    """Median (dx, dy) motion in px from frame a to frame b for each box (in
    frame-a coordinates), or None where too few points tracked reliably."""
    starts, owners = [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Shrink the box 15% per side to stay off the road around the vehicle
        mx, my = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        xa, ya = max(int(x1 + mx), 0), max(int(y1 + my), 0)
        xb, yb = int(x2 - mx), int(y2 - my)
        if xb - xa < 4 or yb - ya < 4:
            continue
        pts = cv2.goodFeaturesToTrack(gray_a[ya:yb, xa:xb], max_points, 0.01, 2, blockSize=3)
        if pts is not None:
            starts.append(pts.reshape(-1, 2) + (xa, ya))
            owners += [i] * len(pts)
    motions = [None] * len(boxes)
    if not starts:
        return motions
    p0 = np.concatenate(starts).astype(np.float32).reshape(-1, 1, 2)
    lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, p0, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, p1, None, **lk)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - p0).reshape(-1, 2), axis=1) < 1.0)
    moves = (p1 - p0).reshape(-1, 2)
    owners = np.array(owners)
    dist = np.linalg.norm(moves, axis=1)
    for i in range(len(boxes)):
        sel = good & (owners == i)
        if sel.sum() < 3:
            continue
        # Boxes also catch bits of static road, lights and glare. If a good share
        # of the points move, the vehicle is moving and the still ones are
        # background; otherwise the vehicle itself is (nearly) still.
        moving = sel & (dist >= STOPPED_PX)
        use = moving if moving.sum() >= max(2, 0.4 * sel.sum()) else sel
        motions[i] = np.median(moves[use], axis=0)
    return motions


def speed_mph(box, cls, motion, dt):
    """Ground speed in mph from a box's pixel motion over dt seconds, or None."""
    if motion is None or dt <= 0:
        return None
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < MIN_BOX_PX:
        return None
    width_m, length_m, height_m = DIMENSIONS[cls]
    dist_px = math.hypot(*motion)
    if dist_px < STOPPED_PX:
        return 0.0
    ux, uy = abs(motion[0]) / dist_px, abs(motion[1]) / dist_px
    px_per_m = w / (ux * length_m + uy * width_m)                     # box width is length when crossing, width head-on
    own_length_px = ux * w + uy * max(h - height_m * px_per_m, 0.25 * h)  # its length along the motion
    mph = float(dist_px / own_length_px * length_m / dt * MPS_TO_MPH)
    return mph if mph <= MAX_MPH else None


def measure(gray_a, gray_b, detections, dt):
    """Speeds (mph or None) for detections [(box, cls), ...] found in frame a,
    given frame b taken dt seconds later (negative dt: b is earlier)."""
    motions = box_motions(gray_a, gray_b, [box for box, _ in detections])
    if dt < 0:
        motions = [None if m is None else -m for m in motions]
    return [speed_mph(box, cls, m, abs(dt)) for (box, cls), m in zip(detections, motions)]


def summarize(speeds):
    """Stats for one frame's speeds: median of moving vehicles, counts."""
    known = [s for s in speeds if s is not None]
    moving = [s for s in known if s >= 3]
    return dict(median_mph=float(np.median(moving)) if moving else None,
                moving=len(moving), stopped=len(known) - len(moving), measured=len(known))
