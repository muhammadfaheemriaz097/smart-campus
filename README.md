# AI-Based Smart Campus Safety & Monitoring System

A Python application that watches campus video feeds and turns them into attendance records, headcounts and security alerts, all shown on one Streamlit dashboard.

## Features

| # | Module | What it does |
|---|--------|--------------|
| 1 | Person Detection | Detects and counts people in each frame (YOLO) |
| 2 | Entry/Exit Monitoring | Counts people crossing a virtual line, by direction |
| 3 | Attendance & Verification | Face recognition (InsightFace) with a head-turn liveness check, logged once per person |
| 4 | Restricted-Area Monitoring | Alerts when someone enters a zone outside permitted hours |
| 5 | Crowd Monitoring | Compares live occupancy with each area's capacity |
| 6 | Dashboard | Live video, statistics, alerts and history, with role-based login |

## How it works

1. Each camera has one worker: **detect → track (ByteTrack) → run checks** (identity, line crossing, zones, occupancy).
2. Faces are matched **once per tracked person**, not every frame.
3. Events are saved to **SQLite (WAL mode)**.
4. The Streamlit dashboard only **reads** from the database, so a slow screen never slows detection.

## Tech stack

Python 3.11 · OpenCV · Ultralytics YOLO · ByteTrack · InsightFace (ArcFace) · Streamlit · SQLite · Pandas

## Project structure

```
smart_campus/
├── app.py                # Streamlit dashboard entry point
├── worker.py             # Per-camera detection/tracking worker
├── modules/
│   ├── detection.py
│   ├── counting.py       # entry/exit line
│   ├── zones.py          # restricted areas
│   ├── crowd.py          # capacity alerts
│   └── attendance.py     # face recognition + liveness
├── data/                 # SQLite DB, encrypted embeddings (not committed)
├── config/               # cameras, zones, capacity limits
├── requirements.txt
└── README.md
```

> Adjust this tree to match your actual folders.

## Installation

```bash
git clone https://github.com/muhammadfaheemriaz097/<smart-campus>.git
cd <smart-campus>
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

1. **Configure cameras** (webcam, iVCam phone camera, IP camera or a video file) in `config/`.
2. **Draw rules** per camera: entry/exit line, restricted zones with permitted hours, capacity limits.
3. **Enrol people** (only after written consent): capture face photos → encrypted embeddings → photos deleted.
4. **Start the worker:**
   ```bash
   python worker.py
   ```
5. **Start the dashboard:**
   ```bash
   streamlit run app.py
   ```
6. Log in with your role: `admin`, `security` or `teacher`.

## Performance targets

| Metric | Target |
|--------|--------|
| Detection speed | ≥ 15 FPS (720p) |
| Entry/exit count error | ≤ 10% |
| Face match rate | ≥ 95% |
| Alert delay | ≤ 3 seconds |
| Crowd count error | ≤ 15% |

## Privacy & ethics

- Enrolment is voluntary, with written consent; manual attendance is always available.
- Only encrypted embeddings are stored; enrolment photos are deleted.
- Crowd and entry/exit counts are anonymous.
- Role-based access and an audit log for identity lookups.
- A failed face match never blocks entry on its own; a human override is always available.
- Video clips are deleted after a fixed retention period (e.g. 30 days).

## Limitations

- Small or turned faces in CCTV reduce recognition accuracy.
- Dense crowds can cause undercounting.
- Low-light and backlit scenes lower accuracy.
- Prototype for supervised pilot use; not a replacement for security staff.

## Roadmap

- Phase 2: campus-wide rollout and re-identifying people across cameras
- Mobile app for security staff
- Larger database for many cameras

