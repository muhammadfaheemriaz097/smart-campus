import csv
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from db import init_db, log_event

VIDEO_SOURCE = 1
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.5
TRACKER = "bytetrack.yaml"

# Zone corners as fractions of the frame (x, y). This one is the right side.
ZONE_FRACTIONS = [(0.55, 0.15), (0.95, 0.15), (0.95, 0.95), (0.55, 0.95)]

# The zone is open from 08:00 to 17:00 and restricted outside those hours.
PERMITTED_START, PERMITTED_END = 8, 17
ALWAYS_RESTRICTED = True          # True for testing; set False for the real rule

DWELL_SECONDS = 1.0               # must stay inside this long before alerting
ALERT_BANNER_SECONDS = 3.0


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
os.makedirs("data/alerts", exist_ok=True)
LOG_FILE = "data/alerts_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["time", "track_id", "seconds_to_alert", "snapshot"])


def is_restricted_now():
    if ALWAYS_RESTRICTED:
        return True
    hour = datetime.now().hour
    return not (PERMITTED_START <= hour < PERMITTED_END)


model = YOLO(MODEL_NAME)
cap = LatestFrame(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {VIDEO_SOURCE}")

entered_at = {}       # track ID -> time it entered the zone
alerted = set()       # track IDs that already triggered an alert
alert_count = 0
banner_until = 0.0
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
    zone = np.array([(int(x * w), int(y * h)) for x, y in ZONE_FRACTIONS], np.int32)
    restricted = is_restricted_now()

    results = model.track(
        frame, classes=[0], conf=CONFIDENCE, imgsz=416,
        persist=True, tracker=TRACKER, verbose=False,
    )
    boxes = results[0].boxes
    now = time.time()
    inside_ids = set()

    for box in boxes:
        if box.id is None:
            continue
        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) < 40 or (y2 - y1) < 80:
            continue

        foot = ((x1 + x2) // 2, y2)
        inside = cv2.pointPolygonTest(zone, foot, False) >= 0
        color = (0, 255, 0)

        if inside and restricted:
            inside_ids.add(track_id)
            entered_at.setdefault(track_id, now)
            color = (0, 0, 255)
            waited = now - entered_at[track_id]
            if waited >= DWELL_SECONDS and track_id not in alerted:
                alerted.add(track_id)
                alert_count += 1
                banner_until = now + ALERT_BANNER_SECONDS
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap = f"data/alerts/alert_{stamp}_id{track_id}.jpg"
                cv2.imwrite(snap, frame)
                with open(LOG_FILE, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [datetime.now().isoformat(timespec="seconds"),
                         track_id, round(waited, 2), snap])
                log_event("RESTRICTED_ZONE", track_id=track_id,
                          detail=f"alert after {waited:.2f}s", snapshot=snap)
                print(f"ALERT: ID {track_id} in restricted zone")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, foot, 5, (255, 0, 0), -1)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # forget people who left the zone so they can trigger again later
    for tid in list(entered_at):
        if tid not in inside_ids:
            del entered_at[tid]
            alerted.discard(tid)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], (0, 0, 255) if restricted else (0, 200, 0))
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    cv2.polylines(frame, [zone], True, (0, 255, 255), 2)

    if now < banner_until:
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, "ALERT: RESTRICTED AREA ENTRY", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    status = "RESTRICTED" if restricted else "OPEN"
    cv2.putText(frame, f"Zone: {status}   Alerts: {alert_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"People: {len(boxes)}   FPS: {fps:.1f}", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    from live import publish_frame
    publish_frame(frame, "zone")
    cv2.imshow("Stage 3 - Restricted zone", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()