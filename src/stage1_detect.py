import time
import cv2
from ultralytics import YOLO

VIDEO_SOURCE = 0   # use 0 for webcam
MODEL_NAME = "yolov8n.pt"            # small, fast; downloads automatically
CONFIDENCE = 0.4

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {VIDEO_SOURCE}")

prev_time = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # detect + track people only (class 0 = person)
    results = model.track(
        frame, classes=[0], conf=0.3,
        persist=True, tracker="bytetrack.yaml", verbose=False,
    )

    boxes = results[0].boxes
    count = len(boxes)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # smoothed FPS
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    cv2.putText(frame, f"People: {count}   FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("Stage 1 - Person detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()