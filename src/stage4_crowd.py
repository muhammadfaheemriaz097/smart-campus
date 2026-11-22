import csv
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from db import init_db, log_event, log_occupancy

VIDEO_SOURCE = 1
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.5
TRACKER = "bytetrack.yaml"

# Area to monitor, as fractions of the frame. This one is the whole frame.
AREA_FRACTIONS = [(0.02, 0.02), (0.98, 0.02), (0.98, 0.98), (0.02, 0.98)]
CAPACITY =2                  # for a solo test, set 0 so one person triggers it

OVER_SECONDS = 2.0                # must stay over capacity this long before alert
CLEAR_SECONDS = 2.0               # must be back under capacity this long to clear
LOG_EVERY_SECONDS = 1.0           # how often to save occupancy


class LatestFrame:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.running = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        return (self.frame is not None), self.frame

    def release(self):
        self.running = False
        self.cap.release()


init_db()
os.makedirs("data", exist_ok=True)
OCC_FILE = "data/occupancy_log.csv"
ALERT_FILE = "data/crowd_alerts.csv"
for path, header in [(OCC_FILE, ["time", "occupancy", "capacity"]),
                     (ALERT_FILE, ["time", "event", "occupancy", "capacity"])]:
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def log_row(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


model = YOLO(MODEL_NAME)
cap = LatestFrame(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {VIDEO_SOURCE}")

alert_active = False
over_since = None
under_since = None
alert_count = 0
last_log = 0.0
prev_time = time.time()
fps = 0.0
last_frame = None

while True:
    ok, frame = cap.read()
    if not ok or frame is last_frame:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    last_frame = frame

    h, w = frame.shape[:2]
    area = np.array([(int(x * w), int(y * h)) for x, y in AREA_FRACTIONS], np.int32)

    results = model.track(
        frame, classes=[0], conf=CONFIDENCE, imgsz=416,
        persist=True, tracker=TRACKER, verbose=False,
    )
    boxes = results[0].boxes
    now = time.time()
    occupancy = 0

    for box in boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) < 40 or (y2 - y1) < 80:
            continue
        foot = ((x1 + x2) // 2, y2)
        inside = cv2.pointPolygonTest(area, foot, False) >= 0
        if inside:
            occupancy += 1
        color = (0, 255, 0) if inside else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, foot, 5, (255, 0, 0), -1)
        cv2.putText(frame, f"ID {int(box.id[0])}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # over-capacity logic with a delay both ways, so flicker doesn't spam alerts
    if occupancy > CAPACITY:
        under_since = None
        over_since = over_since or now
        if not alert_active and now - over_since >= OVER_SECONDS:
            alert_active = True
            alert_count += 1
            log_row(ALERT_FILE, [datetime.now().isoformat(timespec="seconds"),
                                 "OVER_CAPACITY", occupancy, CAPACITY])
            log_event("OVER_CAPACITY", detail=f"{occupancy}/{CAPACITY}")
            print(f"ALERT: {occupancy} people, capacity {CAPACITY}")
    else:
        over_since = None
        if alert_active:
            under_since = under_since or now
            if now - under_since >= CLEAR_SECONDS:
                alert_active = False
                under_since = None
                log_row(ALERT_FILE, [datetime.now().isoformat(timespec="seconds"),
                                     "CLEARED", occupancy, CAPACITY])
                log_event("CLEARED", detail=f"{occupancy}/{CAPACITY}")

    if now - last_log >= LOG_EVERY_SECONDS:
        log_row(OCC_FILE, [datetime.now().isoformat(timespec="seconds"),
                           occupancy, CAPACITY])
        log_occupancy(occupancy, CAPACITY)
        last_log = now

    cv2.polylines(frame, [area], True, (0, 255, 255), 2)

    if alert_active:
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, f"OVERCROWDED: {occupancy} / {CAPACITY}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    status_color = (0, 0, 255) if occupancy > CAPACITY else (0, 150, 0)
    cv2.putText(frame, f"Occupancy: {occupancy} / {CAPACITY}   Alerts: {alert_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    from live import publish_frame
    publish_frame(frame,"crowd")
    cv2.imshow("Stage 4 - Crowd monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()