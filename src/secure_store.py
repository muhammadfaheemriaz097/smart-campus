import os
import pickle

from cryptography.fernet import Fernet

KEY_FILE = "data/secret.key"
DB_FILE = "data/faces.enc"


def _fernet():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(KEY_FILE):
        with open(KEY_FILE, "wb") as f:
            f.write(Fernet.generate_key())
    with open(KEY_FILE, "rb") as f:
        return Fernet(f.read())


def load_faces():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "rb") as f:
        return pickle.loads(_fernet().decrypt(f.read()))


def save_faces(db):
    with open(DB_FILE, "wb") as f:
        f.write(_fernet().encrypt(pickle.dumps(db)))