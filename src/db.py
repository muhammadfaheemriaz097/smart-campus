import sqlite3
from datetime import datetime
from pathlib import Path

# Absolute path based on this file's location, so it works locally and on Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)   # creates the folder if missing
DB_PATH = str(DATA_DIR / "campus.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY, name TEXT, class_name TEXT, date TEXT, time TEXT,
        UNIQUE(name, class_name, date));
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY, time TEXT, camera TEXT, type TEXT, track_id INTEGER,
        detail TEXT, snapshot TEXT, review TEXT DEFAULT 'open',
        reviewed_by TEXT, reviewed_at TEXT);
    CREATE TABLE IF NOT EXISTS occupancy (
        id INTEGER PRIMARY KEY, time TEXT, camera TEXT, occupancy INTEGER, capacity INTEGER);
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY, salt TEXT, pw_hash TEXT, role TEXT);
    CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY, time TEXT, username TEXT, action TEXT, detail TEXT);
    """)
    conn.commit()
    conn.close()


def now_str():
    return datetime.now().isoformat(timespec="seconds")


def log_event(type, camera="cam1", track_id=None, detail="", snapshot=""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO events (time, camera, type, track_id, detail, snapshot) "
        "VALUES (?,?,?,?,?,?)",
        (now_str(), camera, type, track_id, detail, snapshot))
    conn.commit()
    conn.close()


def log_occupancy(occupancy, capacity, camera="cam1"):
    conn = get_conn()
    conn.execute(
        "INSERT INTO occupancy (time, camera, occupancy, capacity) VALUES (?,?,?,?)",
        (now_str(), camera, occupancy, capacity))
    conn.commit()
    conn.close()
import hashlib, os

def _hash(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), 200_000
    ).hex()

def seed_users(users):
    """users = list of (username, password, role). Only creates missing users."""
    conn = get_conn()
    for username, password, role in users:
        exists = conn.execute(
            "SELECT 1 FROM users WHERE username=?", (username,)
        ).fetchone()
        if not exists:
            salt = os.urandom(16).hex()
            conn.execute(
                "INSERT INTO users VALUES (?,?,?,?)",
                (username, salt, _hash(password, salt), role),
            )
    conn.commit()
    conn.close()   
