import os
import time

import cv2

LIVE_DIR = "data/live"
_last = {}


def publish_frame(frame, name, every=0.5, width=640):
    """Save the latest annotated frame for the dashboard (overwrites the same file)."""
    now = time.time()
    if now - _last.get(name, 0) < every:
        return
    _last[name] = now
    try:
        os.makedirs(LIVE_DIR, exist_ok=True)
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        tmp = os.path.join(LIVE_DIR, f"{name}.tmp")
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, os.path.join(LIVE_DIR, f"{name}.jpg"))   # swap in one step
    except OSError:
        pass    # the dashboard may be reading the file this instant; skip this frame