#!/usr/bin/env python3
"""
Vehicle detection on traffic cameras, on one live page:

  - Houston TranStar snapshots (a new still every ~2m40s per camera)
  - Baton Rouge live video (Louisiana DOTD streams, the ones 511la.org plays)

The page's switch picks a detector preset (see detectors.py for what each
runs and how it scored on the hand-labeled frames in eval/) without
restarting. "ensemble" fuses YOLO11x and RT-DETR-x and is the default,
except with the incident check on a Mac with under 12 GB of memory, where
the app starts on "yolo11s-hd" (the two together froze an 8 GB Mac).

    .venv/bin/python transtar_feed.py
    .venv/bin/python transtar_feed.py 624 623 503 310 --la-cameras 007 009 001 025
    .venv/bin/python transtar_feed.py --no-live
    .venv/bin/python transtar_feed.py --replay anomaly/clips/tadb/videox_test_T1.mp4   # your own test clip

Then open http://localhost:8500. TranStar results arrive over one Server-Sent
Events stream (/events) and live video over another (/live).

TranStar: every fetch is logged (first / switched / same / error) to the page
and to fetch_log.csv, with the average time between switches. The DeepSORT
tracking and line counting from the thepythoncode.com YOLOv8 article can't run
on stills 2m40s apart, so counts are per snapshot ("vehicles in view").

Speed (rough, see speed.py): live video compares each detected frame with the
one 0.2 s earlier; TranStar cameras that post 6-frame bursts compare frames 1
and 2 (~0.4 s apart). Single-still TranStar cameras have no speed.

Incident check (see incidents.py; scores in anomaly/): a local vision-language
model looks at each running live camera's last 4 frames (1 s apart) every few
seconds, and at each new TranStar snapshot, and flags crashes, stopped
vehicles, people on foot in the road, wrong-way drivers and debris.
--incidents off skips it (and the ~2 GB it takes).
"""

import argparse
import base64
import csv
import json
import os
import re
import threading
import time
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# The DOTD stream server returns empty playlists to FFmpeg's default user agent
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "user_agent;Mozilla/5.0")

import cv2
import numpy as np
import requests
import speed
from detectors import PRESETS, PresetDetector
from incidents import FRAMES as INCIDENT_FRAMES, GAP_S as INCIDENT_GAP_S, MODELS as INCIDENT_MODELS

HERE = Path(__file__).parent
HEADERS = {"User-Agent": "Mozilla/5.0"}

SNAPSHOT_URL = "https://traffic.houstontranstar.org/snapshots/cctv/{}.jpg"
CAMERA_LIST_URL = "https://traffic.houstontranstar.org/data/layers/cctvSnapshots_out.js"
# 610 West Loop @ Richmond and @ Westheimer (dense), 69 SW @ 288, 10 Katy @ Heights (close-up)
DEFAULT_CAMERAS = ["624", "623", "503", "310"]

LA_STREAM_URL = "https://ITSStreamingBR.dotd.la.gov/public/br-cam-{}.streams/playlist.m3u8"
DEFAULT_LA_CAMERAS = ["007", "009", "001", "025"]
# From the captions burned into each stream. 511LA's official camera list
# (with names) needs a free developer key: https://www.511la.org/developers/doc
LA_CAMERA_NAMES = {
    "001": "I-12 @ Drusilla Ln", "003": "I-12 East of 10/12", "005": "I-10/I-12 Split Cam 2",
    "007": "I-10 @ College Dr", "009": "I-10 at Perkins", "011": "I-10/I-110",
    "013": "I-10 @ Florida St Cam 1", "015": "I-10 at LA1", "017": "I-110 @ Gov. Mansion Cam 1",
    "019": "I-10 at Essen EB", "021": "I-10 at Picardy", "023": "I-12 at Airline Hwy NB",
    "025": "US 61 at Goodwood", "027": "US 61 Coursey at Bluebonnet", "033": "Florida Blvd at Stevendale",
    "039": "I-12 at Sherwood #2",
}

# Gap between frames 1 and 2 of a TranStar burst. The frames' burned-in clock
# (whole seconds) puts it at 0.33-0.5 s, so speeds there carry +-25% from this alone.
BURST_GAP_S = 0.4
# Live speeds compare each detected frame with the one ~0.2 s earlier (exact stream timestamps)
LIVE_SPEED_GAP_S = 0.2

# P(incident) at or above this raises an alert. Picked on the development clips (10% of normal
# ones flagged); on the held-out test clips it caught 79% of incidents with no false alarms. See anomaly/README.md.
INCIDENT_ALERT = 0.7
# One check over the line is a "possible" incident; it's confirmed once 2 of the camera's last 3 checks are.
# Real incidents stay in view for many minutes, while false alarms are usually a one-check blip (e.g.
# headlight glare at a night intersection: 1 of 80 night checks, never twice in a row).
INCIDENT_CONFIRM = (2, 3)


def incident_state(recent):
    """recent: the camera's latest P(incident) values, oldest first -> "confirmed", "possible" or "clear"."""
    need, of = INCIDENT_CONFIRM
    if sum(p >= INCIDENT_ALERT for p in recent[-of:]) >= need:
        return "confirmed"
    return "possible" if recent and recent[-1] >= INCIDENT_ALERT else "clear"

# Box colors, shared with the page legend
CLASS_COLORS = {"car": "#4cc9f0", "truck": "#ffb703", "bus": "#f72585", "motorcycle": "#80ed99"}


# ---------------------------------------------------------------- detectors

def hex_to_bgr(color):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def draw_detections(frame, dets, scale=2):
    """Class-colored boxes on a 2x-upscaled copy, so they stay crisp on these small frames.
    Text labels are drawn by the page instead, so they can be toggled."""
    out = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (round(x1 * scale), round(y1 * scale)), (round(x2 * scale), round(y2 * scale)),
                      hex_to_bgr(CLASS_COLORS[d["cls"]]), 2)
    return out


def add_speeds(stats, dets, gray_a, gray_b, dt):
    """Estimate each detection's speed from frame a (where it was detected) to
    frame b, dt seconds later (negative: earlier). Fills d["mph"] and stats["speed"]."""
    for d, mph in zip(dets, speed.measure(gray_a, gray_b, [(d["box"], d["cls"]) for d in dets], dt)):
        d["mph"] = mph
    stats["speed"] = speed.summarize([d["mph"] for d in dets])


def dets_json(dets, frame):
    """Compact per-box data for the page's text labels: [x1, y1, x2, y2, class, conf, mph]
    with coordinates as fractions of the frame."""
    h, w = frame.shape[:2]
    return [[round(d["box"][0] / w, 4), round(d["box"][1] / h, 4), round(d["box"][2] / w, 4), round(d["box"][3] / h, 4),
             d["cls"], round(d["conf"], 2), None if d.get("mph") is None else round(d["mph"], 1)] for d in dets]


class Detector:
    """Timing, stats and drawing around one detection preset (see detectors.py).
    Not thread-safe: give each thread its own instance."""

    key = "yolo"  # the page's name for the detection slot

    def __init__(self, preset):
        self.preset = PresetDetector(preset)
        self.name, self.detail = self.preset.name, self.preset.detail
        self.preset.detect(np.zeros((240, 320, 3), np.uint8))  # warm-up, untimed
        self.runs, self.total_ms = 0, 0.0

    def __call__(self, frame):
        """Returns (annotated frame, stats, detections as dicts with box, cls, conf)."""
        started = time.perf_counter()
        boxes, names, confs = self.preset.detect(frame)
        ms = (time.perf_counter() - started) * 1000
        self.runs += 1
        self.total_ms += ms
        dets = [dict(box=b, cls=n, conf=c) for b, n, c in zip(boxes, names, confs)]
        stats = dict(
            vehicles=len(names),
            counts={name: names.count(name) for name in CLASS_COLORS},
            mean_conf=float(np.mean(confs)) if len(confs) else None,
            inference_ms=ms,
        )
        return draw_detections(frame, dets), stats, dets

    @property
    def avg_ms(self):
        return self.total_ms / self.runs if self.runs else None


class ModelChoice:
    """Which detector preset is in use; the page's switch changes it. Detection
    threads notice through `version`; `wake` wakes the TranStar loop."""

    def __init__(self, active, wake):
        self.active, self.wake = active, wake
        self.version = 0

    def set(self, name):
        if name not in PRESETS:
            raise KeyError(name)
        if name != self.active:
            self.active = name
            self.version += 1
            self.wake.set()

    def to_json(self):
        return dict(active=self.active, options=[dict(id=k, name=v["name"]) for k, v in PRESETS.items()])


class ViewState:
    """Which cameras run. "one" (the default): only the camera being viewed is
    fetched, decoded and run through the detector; the page shows the rest as
    placeholders, because running the ensemble on every camera at once makes
    viewing laggy. "all": every camera runs, for collecting data and comparing.
    Cameras are keyed "tt:<id>" (TranStar) and "br:<id>" (Baton Rouge)."""

    def __init__(self, mode, active, keys, wake):
        self.mode, self.active, self.keys, self.wake = mode, active, keys, wake
        self.version = 0
        self.listeners = []  # called after every change

    def set(self, mode=None, active=None):
        if mode not in (None, "one", "all") or (active is not None and active not in self.keys):
            raise KeyError(mode if mode not in (None, "one", "all") else active)
        changed = (mode or self.mode) != self.mode or (active or self.active) != self.active
        self.mode, self.active = mode or self.mode, active or self.active
        if changed:
            self.version += 1
            for listener in self.listeners:
                listener()
            self.wake.set()

    def is_on(self, key):
        return self.mode == "all" or key == self.active

    def to_json(self):
        return dict(mode=self.mode, active=self.active)


class SwitchableDetector:
    """Calls go to the Detector for whichever preset the ModelChoice says is
    active. Presets load on first use, and the one switched away from is
    unloaded: the Ensemble alone takes ~2 GB. One instance serves every
    thread (TranStar and live), one frame at a time, since Ultralytics models
    aren't thread-safe and a second copy would double the memory."""

    key = Detector.key

    def __init__(self, choice):
        self.choice = choice
        self._detectors = {}
        self._lock = threading.RLock()

    @property
    def current(self):
        with self._lock:
            preset = self.choice.active
            if preset not in self._detectors:
                if self._detectors:
                    self._detectors.clear()
                    import gc, torch
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                self._detectors[preset] = Detector(preset)
            return self._detectors[preset]

    def __call__(self, frame):
        with self._lock:
            return self.current(frame)

    name = property(lambda self: self.current.name)
    detail = property(lambda self: self.current.detail)
    avg_ms = property(lambda self: self.current.avg_ms)

    def to_json(self):
        return dict(id=self.choice.active, name=self.name, detail=self.detail)


def encode_jpeg(frame, quality=90):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# ---------------------------------------------------------------- TranStar snapshots

def fetch_transtar_frame(url, etag=None):
    """Return (BGR frame, camera snapshot time in epoch seconds or None, etag),
    or None if the camera hasn't published a new snapshot since `etag`.
    Raises on failure."""
    headers = dict(HEADERS)
    if etag:
        headers["If-None-Match"] = etag  # unchanged image -> 304, zero bytes
    response = requests.get(url, timeout=15, headers=headers)
    if response.status_code == 304:
        return None
    response.raise_for_status()

    # Decode the raw bytes into an OpenCV image (BGR)
    frame = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        # Bad camera IDs redirect to an HTML page instead of 404ing
        raise requests.RequestException(f"not an image (got {response.headers.get('Content-Type')})")

    try:
        snapshot_at = parsedate_to_datetime(response.headers["Last-Modified"]).timestamp()
    except (KeyError, TypeError, ValueError):
        snapshot_at = None
    return frame, snapshot_at, response.headers.get("ETag")


def fetch_camera_list():
    """{snapshot filename: (camera name, frames per snapshot)} from TranStar's
    camera list, or {} if unavailable. Cameras with 6 frames post a short burst."""
    try:
        js = requests.get(CAMERA_LIST_URL, timeout=15, headers=HEADERS).text
    except requests.RequestException:
        return {}
    cameras = {}
    for args in re.findall(r"new CctvCamera\((.*?)\);", js):
        fields = re.findall(r"'((?:[^'\\]|\\.)*)'", args)
        if len(fields) >= 10:
            # CctvCamera(name, monitor, roadway, location, lat, lng, dir, path, validimg, frameCount, framePauseMs)
            cameras[fields[7]] = (fields[0], int(fields[9]) if fields[9].isdigit() else 1)
    return cameras


def fetch_burst_frame(url, snapshot_at, tries=3):
    """The second frame of a burst camera's snapshot (<id>-2.jpg, ~BURST_GAP_S
    after the first), once TranStar has written it. None if it doesn't show up."""
    for attempt in range(tries):
        try:
            frame, frame_at, _ = fetch_transtar_frame(url.replace(".jpg", "-2.jpg"))
        except requests.RequestException:
            return None
        # Frames are written within ~2 s of each other; an older file is the previous burst
        if snapshot_at is None or frame_at is None or snapshot_at - 1 <= frame_at <= snapshot_at + 5:
            return frame
        time.sleep(1.5)
    return None


class FeedState:
    """Latest TranStar images + status, shared between the fetch thread and HTTP handlers."""

    def __init__(self):
        self._cond = threading.Condition()
        self.images = {}  # URL path -> JPEG bytes, e.g. "/raw/624.jpg", "/det/yolo/624.jpg"
        self.status = {"seq": 0}

    def publish(self, status, images=None):
        with self._cond:
            self.images.update(images or {})
            self.status = {**status, "seq": self.status["seq"] + 1}
            self._cond.notify_all()

    def wait_for_update(self, after_seq, timeout):
        with self._cond:
            self._cond.wait_for(lambda: self.status["seq"] > after_seq, timeout)
            return self.status


def fmt_secs(secs):
    if secs is None:
        return "-"
    secs = round(secs)
    return f"{secs}s" if secs < 60 else f"{secs // 60}m {secs % 60}s"


class SwitchTracker:
    """Records every fetch, whether each camera's image switched, and the time
    between switches. Switch times come from the camera's Last-Modified
    header, so they measure TranStar's real update cadence rather than our
    polling interval (as long as we poll faster than the camera updates)."""

    CSV_FIELDS = ["checked_at", "camera", "result", "snapshot_at", "secs_since_prev_switch", "error"]

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.log = []
        self.gaps = defaultdict(list)       # camera -> seconds between switches
        self.counts = defaultdict(Counter)  # camera -> result counts
        self._last_switch = {}

    def record(self, camera, checked_at, result, snapshot_at=None, error=None):
        """result is "new", "same", or "error". A camera's first image of a run is
        logged as "first", since we can't know when the one before it appeared."""
        gap = None
        if result == "new":
            switched_at = snapshot_at or checked_at
            if camera not in self._last_switch:
                result = "first"
            else:
                result = "switched"
                gap = switched_at - self._last_switch[camera]
                self.gaps[camera].append(gap)
            self._last_switch[camera] = switched_at
        self.counts[camera][result] += 1
        entry = dict(n=len(self.log) + 1, camera=camera, checked_at=checked_at, result=result,
                     snapshot_at=snapshot_at, gap=gap, error=error)
        self.log.append(entry)
        self._append_csv(entry)
        return entry

    def _append_csv(self, entry):
        iso = lambda t: datetime.fromtimestamp(t).isoformat(timespec="seconds") if t else ""
        new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(self.CSV_FIELDS)
            gap = "" if entry["gap"] is None else round(entry["gap"])
            writer.writerow([iso(entry["checked_at"]), entry["camera"], entry["result"],
                             iso(entry["snapshot_at"]), gap, entry["error"] or ""])

    def summary(self, camera=None):
        """Stats for one camera, or all cameras combined."""
        cameras = [camera] if camera else list(self.counts)
        gaps = [g for c in cameras for g in self.gaps[c]]
        counts = sum((self.counts[c] for c in cameras), Counter())
        return dict(
            fetches=sum(counts.values()), switched=counts["switched"],
            same=counts["same"], errors=counts["error"],
            avg_switch_secs=sum(gaps) / len(gaps) if gaps else None,
            min_switch_secs=min(gaps, default=None), max_switch_secs=max(gaps, default=None),
        )

    def recent(self, n=60):
        return self.log[-n:][::-1]

    def describe(self, entry):
        line = f"{entry['camera']}: {entry['result']}"
        if entry["gap"] is not None:
            line += f" ({fmt_secs(entry['gap'])} since previous switch)"
        if entry["error"]:
            line += f": {entry['error']}"
        return line


class Camera:
    """A TranStar snapshot camera and each model's results on its latest snapshot."""

    def __init__(self, spec, camera_list, model_keys):
        self.url = SNAPSHOT_URL.format(spec) if spec.isdigit() else spec
        filename = Path(urlparse(self.url).path).name
        self.id = Path(filename).stem
        self.key = "tt:" + self.id
        self.name, frame_count = camera_list.get(filename, (self.id, 1))
        self.burst = frame_count > 1  # posts a multi-frame burst, so speed is measurable
        self.etag = self.snapshot_at = self.checked_at = self.result = self.error = None
        self.last = None  # (frame, burst frame 2 or None) of the latest snapshot
        self.frames = 0   # new snapshots seen (and model re-runs); cache-busts the image URLs
        self.model_version = None  # ModelChoice.version the current results came from
        # per model: latest snapshot's stats and boxes, vehicles in each snapshot this run,
        # median speed of each snapshot this run
        self.models = {key: dict(stats=None, dets=[], history=[], speeds=[], error=None) for key in model_keys}
        self.incident = None  # latest incident check: dict(p, text, at, ms, model, frames, recent, state) or dict(error)
        self.incident_recent = deque(maxlen=5)  # P(incident) of the latest snapshots

    def to_json(self, tracker):
        def session(counts, speeds):
            return dict(snapshots=len(counts), avg=sum(counts) / len(counts) if counts else None,
                        peak=max(counts, default=None), avg_mph=float(np.mean(speeds)) if speeds else None)
        return dict(
            id=self.id, key=self.key, name=self.name, url=self.url, frame=self.frames, burst=self.burst,
            snapshot_at=self.snapshot_at, checked_at=self.checked_at, result=self.result,
            error=self.error, switching=tracker.summary(self.id), incident=self.incident,
            models={key: dict(stats=m["stats"], dets=m["dets"], error=m["error"],
                              session=session(m["history"], m["speeds"]))
                    for key, m in self.models.items()},
        )


def check_incident(model, frames, gap):
    """Run the incident check; describe what the model sees when it's an alert."""
    started = time.time()
    try:
        p = model.check(frames, gap=gap)
        text = model.describe(frames, gap=gap) if p >= INCIDENT_ALERT else None
    except Exception as e:
        traceback.print_exc()
        return dict(error=f"incident check failed: {e}", at=time.time())
    return dict(p=p, text=text, at=time.time(), ms=(time.time() - started) * 1000, model=model.name, frames=len(frames))


def poll_camera(cam, detectors, tracker):
    """Fetch one camera; on a new snapshot, run every model. Returns (log entry, new images)."""
    cam.checked_at, cam.error, images = time.time(), None, {}
    try:
        result = fetch_transtar_frame(cam.url, cam.etag)
    except Exception as e:
        # Keep serving the last good images; the page shows the error
        entry = tracker.record(cam.id, cam.checked_at, "error", error=str(e))
        cam.error = f"fetch failed: {e}"
        if not isinstance(e, requests.RequestException):
            traceback.print_exc()
    else:
        if result is None:
            entry = tracker.record(cam.id, cam.checked_at, "same", cam.snapshot_at)
        else:
            frame, cam.snapshot_at, cam.etag = result
            entry = tracker.record(cam.id, cam.checked_at, "new", cam.snapshot_at)
            images[f"/raw/{cam.id}.jpg"] = encode_jpeg(frame)
            # Burst cameras: frame 2 (~0.4 s later) shows how far each vehicle moved
            second = fetch_burst_frame(cam.url, cam.snapshot_at) if cam.burst else None
            cam.last = (frame, second)
            for det in detectors:
                run_model(cam, det, images)
            cam.frames += 1
    cam.result = entry["result"]
    return entry, images


def run_model(cam, det, images):
    """Run one model on the camera's latest snapshot, plus speeds from burst frame 2."""
    frame, second = cam.last
    m = cam.models[det.key]
    cam.model_version = det.choice.version
    try:
        annotated, m["stats"], dets = det(frame)
        if second is not None:
            add_speeds(m["stats"], dets, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), BURST_GAP_S)
            if m["stats"]["speed"]["median_mph"] is not None:
                m["speeds"].append(m["stats"]["speed"]["median_mph"])
        m["dets"] = dets_json(dets, frame)
        images[f"/det/{det.key}/{cam.id}.jpg"] = encode_jpeg(annotated)
        m["history"].append(m["stats"]["vehicles"])
        m["error"] = None
    except Exception as e:
        traceback.print_exc()  # a bug in detection, not a network hiccup
        m["error"] = f"detection failed: {e}"


def publish(state, cameras, detectors, tracker, view, next_fetch_at, images=None):
    state.publish(dict(
        next_fetch_at=next_fetch_at, view=view.to_json(),
        model=detectors[0].to_json(), choice=detectors[0].choice.to_json(),
        cameras=[c.to_json(tracker) for c in cameras],
        overall=dict(tracker.summary(), avg_ms={d.key: d.avg_ms for d in detectors}),
        log=tracker.recent(),
    ), images)


def rerun_stale(state, cameras, detectors, tracker, view, next_fetch_at):
    """Re-run the current model on the current snapshot of every camera that's
    on but whose results came from another model (after a switch, or a camera
    that was paused during one), instead of waiting up to ~2m40s for a new one."""
    for cam in cameras:
        if not view.is_on(cam.key) or cam.last is None or cam.model_version == detectors[0].choice.version:
            continue
        images = {}
        for det in detectors:
            run_model(cam, det, images)
        cam.frames += 1  # new image URLs, so the page reloads them
        publish(state, cameras, detectors, tracker, view, next_fetch_at(), images)


def log_line(cam, entry, images, detectors, tracker):
    line = tracker.describe(entry)
    if images:
        parts = []
        for d in detectors:
            s = cam.models[d.key]["stats"]
            if s:
                mph = (s.get("speed") or {}).get("median_mph")
                parts.append(f"{d.name} {s['vehicles']}" + (f" @ {mph:.0f} mph" if mph is not None else ""))
        if cam.incident and "p" in cam.incident:
            parts.append(f"incident {cam.incident['p']:.0%} ({cam.incident['state']})" +
                         (f": {cam.incident['text']}" if cam.incident["text"] else ""))
        line += " -> " + ", ".join(parts)
    return f"[{time.strftime('%H:%M:%S')}] {line}"


def fetch_loop(state, cameras, detectors, tracker, interval, choice, view, wake, incidents=None):
    # TranStar cameras publish a still every ~2m40s, not video. Each camera
    # that's on is checked every `interval` seconds; fetches in between return
    # the same image (HTTP 304), so only new snapshots go through the detector.
    due = {cam.id: 0.0 for cam in cameras}
    next_fetch_at = lambda: min((due[c.id] for c in cameras if view.is_on(c.key)), default=None)
    seen_model = choice.version
    while True:
        for cam in cameras:
            if not view.is_on(cam.key) or due[cam.id] > time.time():
                continue
            due[cam.id] = time.time() + interval
            entry, images = poll_camera(cam, detectors, tracker)
            publish(state, cameras, detectors, tracker, view, next_fetch_at(), images)
            if images and incidents is not None:  # after publishing, so the new snapshot shows right away
                frame, second = cam.last
                cam.incident = check_incident(incidents, [frame] if second is None else [frame, second], BURST_GAP_S)
                if "p" in cam.incident:
                    cam.incident_recent.append(cam.incident["p"])
                    cam.incident.update(recent=list(cam.incident_recent), state=incident_state(list(cam.incident_recent)))
                publish(state, cameras, detectors, tracker, view, next_fetch_at())
            print(log_line(cam, entry, images, detectors, tracker))

        # CRITICAL: respect TranStar's update interval to avoid an IP ban. A model
        # switch or a change of camera wakes us early, but a camera is never
        # fetched again before its interval is up.
        upcoming = next_fetch_at()
        if wake.wait(None if upcoming is None else max(0.0, upcoming - time.time())):
            wake.clear()
            if choice.version != seen_model:
                seen_model = choice.version
                print(f"[{time.strftime('%H:%M:%S')}] switched to {detectors[0].name}")
                for cam in cameras:  # session stats start over with the new model
                    for det in detectors:
                        cam.models[det.key].update(history=[], speeds=[])
            rerun_stale(state, cameras, detectors, tracker, view, next_fetch_at)
            publish(state, cameras, detectors, tracker, view, next_fetch_at())  # the page learns the change


# ---------------------------------------------------------------- Baton Rouge live video

class LiveHub:
    """Latest JPEG + info for each (camera, kind), where kind is "raw" or a
    model key. Each /live client is sent only the slots that changed since it
    last looked, so a slow browser skips frames instead of falling behind."""

    def __init__(self):
        self._lock = threading.Lock()
        self._slots = {}  # (camera, kind) -> (version, base64 JPEG or None, info)

    def put(self, camera, kind, jpeg=None, **info):
        with self._lock:
            version, old_jpeg, _ = self._slots.get((camera, kind), (0, None, None))
            b64 = base64.b64encode(jpeg).decode() if jpeg is not None else old_jpeg
            self._slots[(camera, kind)] = (version + 1, b64, info)

    def changed_since(self, seen):
        with self._lock:
            return {key: slot for key, slot in self._slots.items() if seen.get(key) != slot[0]}


def small_copy(video, height=240):
    """A 240p copy of a replay video in a temp folder, scaled like the anomaly tests
    (area averaging): decoding full-HD video at 25 fps competes with the models for the
    CPU and made playback stutter. The copy keeps the original's name for the page."""
    import tempfile
    cap = cv2.VideoCapture(str(video))
    fps, h = cap.get(cv2.CAP_PROP_FPS) or 25, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if h <= height * 1.5:
        cap.release()
        return video
    # MJPEG at high quality: OpenCV's mp4v encoder is blocky enough to cut the incident model's scores
    out = Path(tempfile.mkdtemp(prefix="replay-")) / (Path(video).stem + ".avi")
    size = (round(height * cap.get(cv2.CAP_PROP_FRAME_WIDTH) / h / 2) * 2, height)  # even width, as in anomaly/
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    writer.release()
    return out


class LiveCamera:
    """One DOTD HLS stream. FFmpeg hands frames over in bursts: ~30 s of
    backlog on connect, then a 10-20 s chunk every ~10 s. A reader thread
    buffers them and a player thread releases each one at its real-time
    moment, so the page sees steady video (15 fps) instead of fast-forward
    bursts and freezes. Detection threads grab whichever frame is current,
    and compare it with a slightly earlier one to measure speed."""

    MAX_BUFFER_S = 15  # enough to ride out the gap between chunks; older frames are skipped
    MAX_BUFFER_FRAMES = 30 * MAX_BUFFER_S  # hard cap, whatever the timestamps say
    REPLAY_AHEAD_S = 3  # a video file is read only this far ahead of playback

    def __init__(self, cam_id, hub, model_keys, replay=None, title=None, expect=None):
        """replay: a video file to play on a loop as if it were this camera (for demos:
        live cameras rarely show a crash), shown at 240p like the DOTD streams. title and
        expect (should the incident check flag it?) come from demo_clips/clips.json."""
        self.id = cam_id
        self.key = "br:" + cam_id
        self.replay, self.expect = replay, expect
        self.name = title or (f"Replay: {Path(replay).stem}" if replay else
                              LA_CAMERA_NAMES.get(cam_id, f"Baton Rouge cam {cam_id}"))
        self.enabled = threading.Event()  # set while the camera is on (see ViewState)
        self.url = str(replay) if replay else LA_STREAM_URL.format(cam_id)
        self.hub = hub
        self._lock = threading.Lock()
        self._buffer = deque()  # (pts seconds, frame) waiting to be shown
        self.status, self.reconnects = "paused", 0
        self.frame, self.seq, self.pts = None, 0, None
        self._shown = deque(maxlen=12)  # (pts, frame) most recently shown, for speed
        self._history = deque(maxlen=INCIDENT_FRAMES)  # (pts, frame) ~INCIDENT_GAP_S apart, for the incident check
        self._recent = {key: deque() for key in model_keys}               # (time, vehicles, moving mph), last 60 s
        self._det_times = {key: deque(maxlen=10) for key in model_keys}   # for detections/second

    def start(self, raw_fps):
        threading.Thread(target=self._read_forever, daemon=True).start()
        threading.Thread(target=self._play_forever, args=(raw_fps,), daemon=True).start()

    def latest(self):
        with self._lock:
            return self.frame, self.seq, self.pts

    def history(self):
        """(newest pts, frames oldest first) for the incident check, once there are enough; else None."""
        with self._lock:
            if len(self._history) < INCIDENT_FRAMES:
                return None
            return self._history[-1][0], [f for _, f in self._history]

    def earlier(self, pts, gap):
        """(pts, frame) of the shown frame closest to `gap` seconds before pts, or None."""
        with self._lock:
            candidates = [(p, f) for p, f in self._shown if p < pts - gap / 2]
        return min(candidates, key=lambda c: abs(c[0] - (pts - gap)), default=None)

    def _read_forever(self):
        while True:
            self.enabled.wait()  # a paused camera holds no connection
            self.status = "connecting"
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000])
            last_pts = None
            with self._lock:
                # A new connection (or a replay starting over) restarts the timestamps; frames left from
                # before would break the time-based trimming below, and the buffer grew without limit
                self._buffer.clear()
            while cap.isOpened() and self.enabled.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                with self._lock:
                    if last_pts is not None and pts < last_pts:
                        self._buffer.clear()  # timestamps restarted
                    self._buffer.append((pts, frame))
                    while pts - self._buffer[0][0] > self.MAX_BUFFER_S or len(self._buffer) > self.MAX_BUFFER_FRAMES:
                        self._buffer.popleft()
                last_pts = pts
                self.status = "live"
                while self.replay and self.enabled.is_set():  # a file reads instantly: keep pace with playback
                    with self._lock:
                        ahead = self._buffer[-1][0] - self._buffer[0][0] if self._buffer else 0
                    if ahead < self.REPLAY_AHEAD_S:
                        break
                    time.sleep(0.05)
            cap.release()
            if not self.enabled.is_set():
                self.status = "paused"
                with self._lock:
                    self._buffer.clear()
                    self._history.clear()
                    self.frame = None
                continue
            if self.replay:
                continue  # end of the file: start it over
            self.reconnects += 1
            self.status = "reconnecting"
            print(f"[{time.strftime('%H:%M:%S')}] Baton Rouge {self.id}: stream dropped, reconnecting")
            time.sleep(5)

    def _play_forever(self, raw_fps):
        offset = None  # wall clock minus stream pts
        shown, fps, window_start, last_push = 0, 0.0, time.time(), 0.0
        sent_status = None  # status last sent without a frame: resend only on change
        while True:
            now = time.time()
            frame = None
            with self._lock:
                if self._buffer:
                    pts = self._buffer[0][0]
                    # Resync on the first frame, after a stall, or after the reader skipped ahead
                    if offset is None or abs(offset + pts - now) > 1:
                        offset = now - pts
                    if offset + pts <= now:
                        pts, frame = self._buffer.popleft()
            if now - window_start >= 2:
                fps, shown, window_start = shown / (now - window_start), 0, now
            if frame is None:
                if self.status != "live" and self.status != sent_status:
                    self.hub.put(self.id, "raw", status=self.status, fps=0, reconnects=self.reconnects)
                    sent_status = self.status
                time.sleep(0.005)
                continue
            # Streams are 352x240 with non-square pixels; the picture is 16:9. Replays come already scaled
            # (small_copy). Scaling matters to the incident model: a plain bilinear shrink of full-HD frames
            # aliases enough to drop a clear crash from 95% to 25% (area averaging, as in anomaly/, doesn't).
            if not self.replay:
                frame = cv2.resize(frame, (round(frame.shape[0] * 16 / 9), frame.shape[0]))
            elif frame.shape[0] != 240:
                frame = cv2.resize(frame, (round(240 * frame.shape[1] / frame.shape[0]), 240), interpolation=cv2.INTER_AREA)
            with self._lock:
                self.frame, self.seq, self.pts = frame, self.seq + 1, pts
                self._shown.append((pts, frame))
                if self._history and not 0 <= pts - self._history[-1][0] < 5:
                    self._history.clear()  # stream restarted or skipped ahead: frames no longer 1 s apart
                if not self._history or pts - self._history[-1][0] >= INCIDENT_GAP_S - 0.03:
                    self._history.append((pts, frame))
            shown += 1
            if now - last_push >= 1 / raw_fps:
                self.hub.put(self.id, "raw", encode_jpeg(frame, 80), status=self.status, fps=fps,
                             reconnects=self.reconnects)
                last_push, sent_status = now, None

    def reset_stats(self, key):
        self._recent[key].clear()
        self._det_times[key].clear()

    def record(self, key, vehicles, dets):
        """Rolling stats for one model's detections on this camera."""
        now = time.time()
        recent, times = self._recent[key], self._det_times[key]
        recent.append((now, vehicles, [d["mph"] for d in dets if (d.get("mph") or 0) >= 3]))
        times.append(now)
        while recent[0][0] < now - 60:
            recent.popleft()
        counts = [v for _, v, _ in recent]
        speeds = [mph for _, _, moving in recent for mph in moving]
        return dict(rate=(len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else None,
                    avg_60s=sum(counts) / len(counts), peak_60s=max(counts),
                    median_mph_60s=float(np.median(speeds)) if speeds else None, speed_samples_60s=len(speeds))


def live_detect_loop(cameras, detector, hub, max_fps, choice):
    """Round-robin over the cameras, always taking each one's newest frame."""
    last_seq, version = {}, choice.version
    while True:
        started = time.time()
        if choice.version != version:  # model switched: rolling stats start over
            version = choice.version
            for cam in cameras:
                cam.reset_stats(detector.key)
        for cam in cameras:
            if not cam.enabled.is_set():
                continue
            frame, seq, pts = cam.latest()
            if frame is None or last_seq.get(cam.id) == seq:
                continue
            last_seq[cam.id] = seq
            try:
                annotated, stats, dets = detector(frame)
                before = cam.earlier(pts, LIVE_SPEED_GAP_S)
                if before is not None:
                    add_speeds(stats, dets, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(before[1], cv2.COLOR_BGR2GRAY), before[0] - pts)
            except Exception as e:
                traceback.print_exc()
                hub.put(cam.id, detector.key, error=f"detection failed: {e}")
                continue
            if choice.version != version:
                break  # switched mid-detection; this result is from the old model
            hub.put(cam.id, detector.key, encode_jpeg(annotated, 80), stats=stats, dets=dets_json(dets, frame),
                    model=detector.name, avg_ms=detector.avg_ms, **cam.record(detector.key, stats["vehicles"], dets))
        # Cap each camera at max_fps detections/second (and don't spin while streams connect)
        time.sleep(max(0.05, 1 / max_fps - (time.time() - started)))


def live_incident_loop(cameras, model, hub, every):
    """Checks each running camera's last few frames every `every` seconds (or as
    fast as the model goes, round-robin, when several cameras are on). A real
    incident stays in view for minutes, so the page also shows how many of the
    last few checks agreed: one alert alone is more likely a false alarm."""
    checked = {}  # camera id -> (time, pts of the newest frame checked)
    recent_p = defaultdict(lambda: deque(maxlen=5))
    while True:
        did = False
        for cam in cameras:
            if not cam.enabled.is_set():
                recent_p.pop(cam.id, None)  # a paused camera starts over
                continue
            recent = cam.history()
            last_at, last_pts = checked.get(cam.id, (0, None))
            if recent is None or recent[0] == last_pts or time.time() - last_at < every:
                continue
            checked[cam.id] = (time.time(), recent[0])
            result = check_incident(model, recent[1], INCIDENT_GAP_S)
            if "p" in result:
                recent_p[cam.id].append(result["p"])
                result.update(recent=list(recent_p[cam.id]), state=incident_state(list(recent_p[cam.id])))
            hub.put(cam.id, "incident", **result)
            did = True
        if not did:
            time.sleep(0.25)


# ---------------------------------------------------------------- web server

def to_json(value):
    # NumPy scalars (e.g. float32 from OpenCV/NumPy math) aren't JSON-serializable
    return json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o))


class Handler(BaseHTTPRequestHandler):
    state = hub = choice = view = None  # set in main()
    page = b""
    log_path = None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", self.page)
        elif path in self.state.images:
            self._send(200, "image/jpeg", self.state.images[path])
        elif path == "/events":
            self._stream_events()
        elif path == "/live":
            self._stream_live()
        elif path == "/fetch_log.csv" and self.log_path.exists():
            self._send(200, "text/csv; charset=utf-8", self.log_path.read_bytes())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        path, _, query = self.path.partition("?")
        params = {k: v[0] for k, v in parse_qs(query).items()}
        try:
            if path == "/model":
                self.choice.set(params.get("name", ""))
                return self._send(200, "application/json", to_json(self.choice.to_json()).encode())
            if path == "/view":
                self.view.set(mode=params.get("mode"), active=params.get("cam"))
                return self._send(200, "application/json", to_json(self.view.to_json()).encode())
        except KeyError:
            return self._send(400, "text/plain", b"unknown model, view mode or camera")
        self._send(404, "text/plain", b"not found")

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _start_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _stream_events(self):
        self._start_stream()
        seen = 0
        try:
            while True:
                status = self.state.wait_for_update(seen, timeout=15)
                if status["seq"] > seen:
                    seen = status["seq"]
                    self.wfile.write(f"data: {to_json(status)}\n\n".encode())
                else:
                    self.wfile.write(b": keepalive\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser tab closed

    def _stream_live(self):
        # Live frames ride one SSE connection as base64 JPEGs: separate MJPEG
        # streams per image would hit the browser's 6-connections-per-host limit
        self._start_stream()
        seen, last_write = {}, time.time()
        try:
            while True:
                changed = self.hub.changed_since(seen)
                if changed:
                    updates = [dict(cam=cam, kind=kind, jpeg=b64, info=info)
                               for (cam, kind), (_, b64, info) in changed.items()]
                    seen.update({key: slot[0] for key, slot in changed.items()})
                    self.wfile.write(f"data: {to_json({'updates': updates})}\n\n".encode())
                    last_write = time.time()
                elif time.time() - last_write > 15:
                    self.wfile.write(b": keepalive\n\n")
                    last_write = time.time()
                time.sleep(1 / 12)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        pass  # keep the console for fetch logs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cameras", nargs="*", default=DEFAULT_CAMERAS,
                        help="TranStar camera IDs (e.g. 624) or snapshot URLs (default: %(default)s)")
    parser.add_argument("--la-cameras", nargs="*", default=DEFAULT_LA_CAMERAS,
                        help="Baton Rouge DOTD camera numbers, br-cam-NNN (default: %(default)s)")
    parser.add_argument("--no-live", action="store_true", help="skip the Baton Rouge live video section")
    parser.add_argument("--replay", nargs="*", default=None, type=Path, metavar="VIDEO",
                        help="video files to play on a loop as test cameras (default: the incident test clips "
                             "in demo_clips/, made by anomaly/make_demo_clips.py)")
    parser.add_argument("--no-demo", action="store_true", help="don't show the incident test clips")
    parser.add_argument("--view", default="one", choices=["one", "all"],
                        help="one: only the camera being viewed runs (default); all: every camera runs")
    parser.add_argument("--interval", type=float, default=60, help="seconds between TranStar fetches (default 60)")
    parser.add_argument("--detector", default="auto", choices=["auto", *PRESETS],
                        help="detector preset to start with; the page can switch. auto: ensemble, or yolo11s-hd "
                             "when the incident check is on and the Mac has under 12 GB of memory (default: auto)")
    parser.add_argument("--live-fps", type=float, default=2,
                        help="max detections/second per live camera (default 2)")
    parser.add_argument("--raw-fps", type=float, default=10, help="live video frames/second sent to the page")
    parser.add_argument("--incidents", default="qwen3vl-2b", choices=[*INCIDENT_MODELS, "off"],
                        help="vision-language model for the incident check, or off (default: %(default)s)")
    parser.add_argument("--incident-every", type=float, default=4,
                        help="seconds between incident checks on each live camera (default 4)")
    parser.add_argument("--port", type=int, default=8500)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your LAN")
    parser.add_argument("--log", type=Path, default=HERE / "fetch_log.csv",
                        help="CSV that every TranStar fetch is appended to (default: fetch_log.csv next to this script)")
    args = parser.parse_args()

    if args.detector == "auto":
        # Ensemble (~2 GB) + incident model (~2 GB) + browser froze an 8 GB Mac; YOLO11s HD takes ~1.5 GB
        memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
        import importlib.util
        incident_check = args.incidents != "off" and importlib.util.find_spec("mlx_vlm") is not None  # Mac only
        args.detector = "yolo11s-hd" if incident_check and memory_gb < 12 else "ensemble"
        if args.detector != "ensemble":
            print(f"Starting on {PRESETS[args.detector]['name']}: with the incident check on, the Ensemble is too much "
                  f"for {memory_gb:.0f} GB of memory (switch on the page, or --incidents off)")
    wake = threading.Event()  # model or view changes wake the TranStar loop
    choice = ModelChoice(args.detector, wake)
    # One detector shared by every thread (see SwitchableDetector). Touching
    # .current loads the starting preset now instead of on the first frame.
    detector = SwitchableDetector(choice)
    detector.current
    transtar_detectors = [detector]
    model_keys = [d.key for d in transtar_detectors]
    camera_list = fetch_camera_list()
    cameras = [Camera(spec, camera_list, model_keys) for spec in args.cameras]
    hub = LiveHub()
    live_cameras = [] if args.no_live else [LiveCamera(n, hub, model_keys) for n in args.la_cameras]
    if args.replay is not None:
        replays = [dict(path=v) for v in args.replay]
    elif not args.no_demo and (HERE / "demo_clips" / "clips.json").exists():
        replays = [dict(path=HERE / "demo_clips" / c["file"], title=c["title"], expect=c["incident"])
                   for c in json.load(open(HERE / "demo_clips" / "clips.json"))]
    else:
        replays = []
    for k, r in enumerate(replays, 1):
        if not Path(r["path"]).exists():
            parser.error(f"--replay: no such file {r['path']}")
        live_cameras.append(LiveCamera(f"replay{k}", hub, model_keys, replay=small_copy(r["path"]),
                                       title=r.get("title"), expect=r.get("expect")))

    all_cameras = cameras + live_cameras
    view = ViewState(args.view, all_cameras[0].key, [c.key for c in all_cameras], wake)
    def apply_live_view():
        for cam in live_cameras:
            (cam.enabled.set if view.is_on(cam.key) else cam.enabled.clear)()
    view.listeners.append(apply_live_view)
    apply_live_view()

    incidents = None
    if args.incidents != "off":
        try:
            from incidents import IncidentModel  # loads MLX; skipped entirely with --incidents off
            print(f"Loading {INCIDENT_MODELS[args.incidents]['name']} for the incident check...")
            incidents = IncidentModel(args.incidents)
        except ImportError as e:
            print(f"No incident check: it needs an Apple-silicon Mac with mlx-vlm installed ({e})")

    tracker = SwitchTracker(args.log)
    state = FeedState()
    threading.Thread(target=fetch_loop, args=(state, cameras, transtar_detectors, tracker, args.interval,
                                              choice, view, wake, incidents), daemon=True).start()
    for cam in live_cameras:
        cam.start(args.raw_fps)
    if live_cameras:
        threading.Thread(target=live_detect_loop, args=(live_cameras, detector, hub, args.live_fps, choice),
                         daemon=True).start()
        if incidents is not None:
            threading.Thread(target=live_incident_loop, args=(live_cameras, incidents, hub, args.incident_every),
                             daemon=True).start()

    Handler.state, Handler.hub, Handler.choice, Handler.view, Handler.log_path = state, hub, choice, view, args.log
    js = lambda value: json.dumps(value).replace("</", "<\\/")  # safe inside <script>
    Handler.page = ((HERE / "page.html").read_text()
                    .replace("{{CLASS_COLORS}}", js(CLASS_COLORS))
                    .replace("{{MODEL}}", js(transtar_detectors[0].to_json()))
                    .replace("{{YOLO_CHOICE}}", js(choice.to_json()))
                    .replace("{{VIEW}}", js(view.to_json()))
                    .replace("{{TRANSTAR_CAMERAS}}", js([dict(id=c.id, key=c.key, name=c.name, burst=c.burst) for c in cameras]))
                    .replace("{{LIVE_CAMERAS}}", js([dict(id=c.id, key=c.key, name=c.name, replay=bool(c.replay),
                                                          expect=c.expect) for c in live_cameras]))
                    .replace("{{INCIDENTS}}", js(dict(model=incidents.name if incidents else None, alert=INCIDENT_ALERT,
                                                      confirm=INCIDENT_CONFIRM,
                                                      every=args.incident_every,
                                                      accuracy=INCIDENT_MODELS[args.incidents]["accuracy"]
                                                      if incidents else "")))
                    ).encode()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}  "
          f"(TranStar: {', '.join(c.name for c in cameras)}; "
          f"Baton Rouge: {', '.join(c.name for c in live_cameras if not c.replay) or 'off'}; "
          f"test clips: {sum(bool(c.replay) for c in live_cameras)}; Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
