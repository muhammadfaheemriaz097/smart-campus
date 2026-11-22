import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import random
import sqlite3
import threading
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO

from secure_store import load_faces

VIDEO_SOURCE = 1
CONFIDENCE = 0.5
THRESHOLD = 0.40          # cosine similarity needed for a match
FACE_EVERY = 2            # check faces often while someone is pending
CLASS_NAME = "CS-101"     # the class session being logged
REQUIRE_LIVENESS = True   # False = mark attendance without the head turn
YAW_CENTER = 12           # degrees: "facing the camera" below this
YAW_TURN = 20             # degrees: "turned" above this
LEFT_SIGN = 1             # if LEFT and RIGHT feel swapped on screen, set to -1


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


db = load_faces()
if not db:
    raise SystemExit("No enrolled people found. Run enroll.py first.")
names = list(db)
matrix = np.array([np.mean(v, axis=0) for v in db.values()])
matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

conn = sqlite3.connect("data/campus.db")
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("""CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY, name TEXT, class_name TEXT, date TEXT, time TEXT,
    UNIQUE(name, class_name, date))""")

# people already verified today are never asked again, even after a restart
today = datetime.now().strftime("%Y-%m-%d")
verified_names = {row[0] for row in conn.execute(
    "SELECT name FROM attendance WHERE class_name = ? AND date = ?",
    (CLASS_NAME, today))}

face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(320, 320))
if "landmark_3d_68" not in face_app.models:
    raise SystemExit("This face model has no head-pose output, so the turn check cannot run.")
model = YOLO("yolov8n.pt")

cap = LatestFrame(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit("Could not open camera")

identified = {}      # track ID -> name
state = {}           # track ID -> challenge state
frame_idx = 0
banner = ""
last_yaw = None
last_frame = None

while True:
    ok, frame = cap.read()
    if not ok or frame is last_frame:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    last_frame = frame
    frame_idx += 1

    results = model.track(frame, classes=[0], conf=CONFIDENCE, imgsz=416,
                          persist=True, tracker="bytetrack.yaml", verbose=False)
    people = []
    for box in results[0].boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) < 40 or (y2 - y1) < 80:
            continue
        people.append((int(box.id[0]), x1, y1, x2, y2))

    pending = [p for p in people if p[0] not in identified]
    if pending and frame_idx % FACE_EVERY == 0:
        for face in face_app.get(frame):
            fx1, fy1, fx2, fy2 = face.bbox
            if face.det_score < 0.6 or (fx2 - fx1) < 50:
                continue
            cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            for tid, x1, y1, x2, y2 in pending:
                if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                    continue

                # direction is chosen ONCE per track and never changes
                st = state.setdefault(tid, {
                    "centered": False, "turned": False, "cand": None,
                    "want": random.choice(["left", "right"])})

                pose = getattr(face, "pose", None)
                if pose is not None:
                    yaw = float(pose[1])
                    last_yaw = yaw
                    direction = "left" if yaw * LEFT_SIGN > 0 else "right"
                    if abs(yaw) < YAW_CENTER:
                        st["centered"] = True
                    elif (abs(yaw) > YAW_TURN and st["centered"]
                          and direction == st["want"]):
                        st["turned"] = True

                sims = matrix @ face.normed_embedding
                best = int(np.argmax(sims))
                if sims[best] >= THRESHOLD:
                    st["cand"] = names[best]

                cand = st["cand"]
                # already verified earlier = no second challenge
                if cand and (st["turned"] or not REQUIRE_LIVENESS
                             or cand in verified_names):
                    identified[tid] = cand
                    verified_names.add(cand)
                    state.pop(tid, None)
                    now = datetime.now()
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO attendance "
                        "(name, class_name, date, time) VALUES (?,?,?,?)",
                        (cand, CLASS_NAME, now.strftime("%Y-%m-%d"),
                         now.strftime("%H:%M:%S")))
                    conn.commit()
                    banner = (f"Attendance marked: {cand}" if cur.rowcount
                              else f"{cand} already marked today")
                    print(banner)
                break

    for tid, x1, y1, x2, y2 in people:
        who = identified.get(tid)
        if who:
            color, label = (0, 255, 0), who
        elif state.get(tid, {}).get("cand"):
            want = state[tid]["want"].upper()
            color, label = (0, 255, 255), f"Turn head {want}"
        else:
            color, label = (0, 165, 255), "Unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if banner:
        cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    if last_yaw is not None:
        cv2.putText(frame, f"Yaw: {last_yaw:.0f} deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        from live import publish_frame
        publish_frame(frame,"attendance")
    cv2.imshow("Stage 5 - Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()