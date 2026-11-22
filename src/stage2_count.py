import threading
import time

import cv2
from ultralytics import YOLO
from live import publish_frame
from db import init_db, log_event

VIDEO_SOURCE = 1                 # your camera number
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.5
TRACKER = "bytetrack.yaml"
LINE_POSITION = 0.5               # 0.5 = middle of the frame
DEADBAND = 8                      # pixels near the line to ignore (reduces jitter)


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
model = YOLO(MODEL_NAME)
cap = LatestFrame(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {VIDEO_SOURCE}")

count_in = 0
count_out = 0
last_side = {}        # track ID -> last confirmed side of the line (+1 or -1)
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
    line_x = int(w * LINE_POSITION)

    results = model.track(
        frame, classes=[0], conf=CONFIDENCE, imgsz=416,
        persist=True, tracker=TRACKER, verbose=False,
    )
    boxes = results[0].boxes

    for box in boxes:
        if box.id is None:
            continue
        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) < 40 or (y2 - y1) < 80:
            continue

        foot_x, foot_y = (x1 + x2) // 2, y2       # bottom-centre of the box
        dist = foot_x - line_x

        if abs(dist) > DEADBAND:
            side = 1 if dist > 0 else -1
            prev = last_side.get(track_id)
            if prev is not None and prev != side:
                if prev == -1 and side == 1:
                    count_in += 1
                    log_event("ENTRY", track_id=track_id)
                else:
                    count_out += 1
                    log_event("EXIT", track_id=track_id)
            last_side[track_id] = side

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (foot_x, foot_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 2)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    cv2.putText(frame, f"IN: {count_in}   OUT: {count_out}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f"People: {len(boxes)}   FPS: {fps:.1f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    from live import publish_frame
    publish_frame(frame, "counting")
    cv2.imshow("Stage 2 - Entry/Exit counting", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):               # reset the on-screen counts (database keeps its rows)
        count_in = count_out = 0
        last_side.clear()

cap.release()
cv2.destroyAllWindows()