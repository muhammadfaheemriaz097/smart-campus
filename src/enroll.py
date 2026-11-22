import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import threading

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from secure_store import load_faces, save_faces

VIDEO_SOURCE = 1
SHOTS = 5


class LatestFrame:
    """Reads the camera in a background thread so you always get the newest frame."""

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


os.makedirs("data", exist_ok=True)
db = load_faces()

name = input("Enter the person's name: ").strip()
app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320, 320))

cap = LatestFrame(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit("Could not open camera")

embeddings = []
faces = []
frame_idx = 0
last_frame = None
status = "Look at the camera, then press SPACE or C"

while len(embeddings) < SHOTS:
    ok, frame = cap.read()

    if ok and frame is not last_frame:
        last_frame = frame
        frame_idx += 1
        if frame_idx % 3 == 1:
            faces = app.get(frame)

        shown = frame.copy()
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(shown, f"{name}: {len(embeddings)}/{SHOTS}   Q = quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(shown, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow("Enrol", shown)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if key in (32, ord("c")) and last_frame is not None:
        found = app.get(last_frame)
        if len(found) == 1:
            embeddings.append(found[0].normed_embedding)
            status = f"Captured {len(embeddings)}/{SHOTS}"
            print(status)
        elif len(found) == 0:
            status = "No face found - move closer, add light"
            print(status)
        else:
            status = "More than one face - only you in view"
            print(status)

cap.release()
cv2.destroyAllWindows()

if embeddings:
    db[name] = embeddings
    save_faces(db)          # encrypted; no photos are ever saved
    print(f"Saved {name}. Enrolled people: {list(db)}")