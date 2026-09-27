# Blindspot CV

The computer-vision part of Blindspot: vehicle detection, rough speeds and incident detection on traffic cameras,
served as a local web page.

- **Houston TranStar camera snapshots** (each camera posts a new still every ~2.5 min)
- **Baton Rouge live video** (Louisiana DOTD cameras, standing in for Houston live video until we have access)
- **Test clips** of real incidents, played on a loop, if you have the `demo_clips/` folder (see below)

On each camera it draws **vehicle boxes** (car / truck / bus / motorcycle), counts them, estimates **rough speeds**,
and runs an **incident check**: a small vision-language model looks at the last few frames and flags crashes,
stalled or stopped vehicles, people on foot in the road, wrong-way drivers and debris.

## Requirements

- **Python 3.12 or newer**
- **~3 GB free disk** for model downloads, **8 GB of RAM** or more
- **The incident check needs an Apple-silicon Mac** (M1/M2/M3/M4). On any other computer the app runs without it
  and prints a note at startup.
- An NVIDIA GPU or an Apple-silicon Mac makes detection much faster; plain CPU works but is slow.

## Quick start

```bash
git clone https://github.com/qian-json/blindspot-cv.git
cd blindspot-cv
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python transtar_feed.py
```

Open **http://localhost:8500**. The first start downloads the models by itself: the YOLO11s detector (~20 MB, into
`weights/`) and, on a Mac, the Qwen3-VL 2B incident model (~1.8 GB, into `~/.cache/huggingface/`). Switching the
page's model to "Ensemble" downloads another ~250 MB. Stop the app with Ctrl+C.

## Using the page

Only one camera runs at a time, to keep things fast: click a camera card to run it. "All cameras" at the top runs
everything at once (heavy). Each card shows the raw image, the detections (boxes only, and with labels), vehicle
counts, speeds and the incident line:

- **No incident seen / below the alert level**: the model's chance that there's an incident
- **Possible incident** (amber): one check at 70% or more, with the model's one-line description
- **Incident** (red outline): 2 of the camera's last 3 checks at 70% or more. False alarms are usually a one-off;
  real incidents stay in view.

## Options

| Flag | What it does |
|---|---|
| `--incidents off` | skip the incident check (saves ~2 GB of memory) |
| `--detector ensemble` / `yolo11s-hd` | starting detector (the page can switch). Default: Ensemble (most accurate, ~0.8 s a frame on an M1 GPU), or YOLO11s HD (~0.08 s) when the incident check runs on a Mac with under 12 GB of memory |
| `--view all` | run every camera at once instead of one |
| `python transtar_feed.py 624 623 503` | which TranStar cameras to show (IDs from houstontranstar.org) |
| `--la-cameras 007 009` | which Baton Rouge cameras to show |
| `--no-live` | skip the Baton Rouge live video |
| `--replay a.mp4 b.mp4` | play your own video files on a loop as test cameras |
| `--no-demo` | hide the test clips |
| `--port 8600` / `--host 0.0.0.0` | serve on another port / on your network |

## Test clips (not in this repo)

The incident test clips (8 short recordings: an overturned truck, a crash scene, a pedestrian on a highway, spilled
cargo, normal congestion that shouldn't be flagged, ...) come from research datasets that can't be published, so
they aren't in this public repo. Ask Jason for the `demo_clips/` folder and put it next to `transtar_feed.py`; the
app picks it up automatically and shows a "Test clips" section. Without it, everything else works the same.

## Files

| File | What it is |
|---|---|
| `transtar_feed.py` | the app: fetches the cameras, runs the models, serves the page and the data streams |
| `page.html` | the web page (a template the app fills in at startup) |
| `detectors.py` | the two vehicle-detector presets and their settings |
| `speed.py` | speed estimation (optical flow, with the vehicle's own size as the ruler) |
| `incidents.py` | the incident check (Qwen3-VL 2B via MLX) |
| `requirements.txt` | pinned package versions (the ones it was tested with) |

## How good it is (measured)

- **Vehicle boxes** (12 hand-labeled camera frames, 287 vehicles): the Ensemble finds 89% of vehicles with 94% of its
  boxes correct; YOLO11s HD 83% / 84%. Misses are mostly tiny far-away cars; signs and roofs sometimes read as trucks.
- **Incidents** (129 fixed traffic-camera clips from public datasets plus our own recordings): on held-out clips it
  caught 79% of incidents with no false alarms on normal traffic. Weakest on a stalled car that looks parked (2 of
  10). On a single still, like most TranStar cameras, it's a bit weaker (about 72%). The one-line descriptions got
  the gist right in 8 of 11 test incidents but can invent details. At night, 1 of 80 checks on normal traffic
  crossed the alert line (headlight glare) and didn't repeat.
- **Speeds** are rough (±30% or worse): treat them as flowing / slow / stopped rather than exact mph. TranStar
  cameras that post single stills have no speed.

## For AI agents working on this code

**How it runs.** `transtar_feed.py` is one process with a few threads:

- `fetch_loop`: polls each running TranStar camera every `--interval` seconds (default 60; HTTP 304 when unchanged),
  runs the detector on each new snapshot (plus burst frame 2 for speed on the cameras that post bursts), then the
  incident check, and publishes the results.
- `LiveCamera` (one per Baton Rouge camera or test clip): a reader thread pulls the HLS stream with OpenCV and a
  player thread releases frames in real time; `live_detect_loop` runs the detector on each camera's newest frame
  (up to `--live-fps`, default 2/s), and `live_incident_loop` runs the incident check on each camera's last 4 frames,
  1 s apart, every `--incident-every` seconds (default 4).
- The HTTP server serves the page, the JPEGs and two Server-Sent Events streams (below).
- `ViewState`: in "one" mode (default) only the camera being viewed is fetched and processed; the others are
  paused. `POST /view?mode=one|all&cam=<key>` changes it (keys look like `tt:624` or `br:007`).

**Getting the data out.** Everything the page shows comes from two SSE streams, so another program can read the
same data from a running app:

- `GET /events` (TranStar): each message is the full state, JSON. `cameras[]` has, per camera: `id`, `key`,
  `name`, `snapshot_at` (epoch s), `incident` (see below), and `models.yolo.stats` = `{vehicles, counts: {car, truck,
  bus, motorcycle}, mean_conf, inference_ms, speed: {median_mph, moving, stopped, measured}}`, plus
  `models.yolo.dets` = `[[x1, y1, x2, y2 (fractions of the image), class, confidence, mph or null], ...]`. Images:
  `/raw/<id>.jpg` and `/det/yolo/<id>.jpg`.
- `GET /live` (Baton Rouge and test clips): each message is `{"updates": [{cam, kind, jpeg, info}]}` with only what
  changed. `kind` is `raw` (the frame; `info`: `status`, `fps`), `yolo` (detections; `info`: `stats` as above, `dets`,
  `rate`, `avg_60s`, `peak_60s`, `median_mph_60s`) or `incident`. `jpeg` is base64 or null.
- `incident` (both streams): `{p, state, text, at, ms, model, frames, recent}`. `p` is P(incident) from the latest
  check, `state` is `confirmed`, `possible` or `clear`, `text` is the model's description (only when `p` >= 0.7),
  `recent` the camera's last few `p` values. Only `confirmed` incidents should drive decisions like routing.

**Things to know before changing it:**

- **Comments mention `anomaly/`, `eval/` and `make_demo_clips.py`**: those are the test harnesses and datasets that
  produced the accuracy numbers above (and the test clips). They stay in Jason's working copy and aren't needed to
  run or change the app.

- **Don't lower TranStar's polling interval**: it only updates every ~2.5 min and hammering it risks an IP ban.
- **TranStar camera locations** (lat/lng, roadway, direction) are in TranStar's camera list
  (`CAMERA_LIST_URL`, parsed by `fetch_camera_list`); the app currently keeps only the name and frame count.
  Baton Rouge cameras only have names.
- **The Baton Rouge streams need a browser User-Agent**, set through `OPENCV_FFMPEG_CAPTURE_OPTIONS` at the top.
- **Memory**: one shared detector serves every thread (`SwitchableDetector`, with a lock; it unloads the preset it
  switches away from). On Apple GPUs, two threads running PyTorch models at once crash, hence `GPU_LOCK` in
  `detectors.py`.
- **The incident model is sensitive to how frames are shrunk**: shrinking full-HD video with plain bilinear
  resizing dropped a clear crash from 95% to 27%. Use area averaging (`cv2.INTER_AREA`), as the code does.
- **The incident check only runs on Apple silicon** (MLX). On other machines `incidents.py` still imports (for its
  constants), `IncidentModel` can't load, and the app runs without it.

## Licenses

The detectors are Ultralytics YOLO11 and RT-DETR (AGPL-3.0) and the incident model is Qwen3-VL 2B (Apache-2.0).
Camera images come from Houston TranStar and Louisiana DOTD's public feeds.
