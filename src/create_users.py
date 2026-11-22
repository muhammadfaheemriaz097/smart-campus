import hashlib
import os

from db import get_conn, init_db

init_db()
username = input("Username: ").strip()
password = input("Password (you will see it as you type): ")
role = input("Role (admin / security / teacher): ").strip().lower()
if role not in ("admin", "security", "teacher"):
    raise SystemExit("Role must be admin, security or teacher")
if not password:
    raise SystemExit("Password cannot be empty")

salt = os.urandom(16).hex()
pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()

conn = get_conn()
conn.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?)", (username, salt, pw_hash, role))
conn.commit()
conn.close()
print(f"User '{username}' saved with role {role}")