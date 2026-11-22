import glob
import os

from db import get_conn, init_db

init_db()

TABLE_GROUPS = {
    "1": ("Events, occupancy and sightings (counts, alerts, movement)", ["events", "occupancy", "sightings"]),
    "2": ("Attendance records", ["attendance"]),
    "3": ("Audit log", ["audit"]),
}
FILE_GROUPS = {
    "4": ("Alert snapshots", ["data/alerts/*.jpg"]),
    "5": ("Live frames", ["data/live/*.jpg"]),
    "6": ("CSV logs and attendance reports", ["data/*.csv"]),
}

conn = get_conn()
print("What is stored now:\n")
for key, (label, tables) in TABLE_GROUPS.items():
    total = sum(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables)
    print(f"  {key}. {label}: {total} rows")
for key, (label, patterns) in FILE_GROUPS.items():
    total = sum(len(glob.glob(p)) for p in patterns)
    print(f"  {key}. {label}: {total} files")

print("\nNot touched: dashboard users, enrolled faces, encryption key, cameras.")
choice = input("\nType the numbers to clear, separated by commas (or 'all'): ").strip().lower()
picked = set(TABLE_GROUPS) | set(FILE_GROUPS) if choice == "all" else {c.strip() for c in choice.split(",")}
picked &= set(TABLE_GROUPS) | set(FILE_GROUPS)
if not picked:
    raise SystemExit("Nothing chosen. Nothing deleted.")

print("\nThis will permanently delete:")
for key in sorted(picked):
    print("  -", (TABLE_GROUPS.get(key) or FILE_GROUPS.get(key))[0])
if input("Type YES to confirm: ").strip() != "YES":
    raise SystemExit("Cancelled. Nothing deleted.")

for key in sorted(picked):
    if key in TABLE_GROUPS:
        for t in TABLE_GROUPS[key][1]:
            conn.execute(f"DELETE FROM {t}")
    else:
        for pattern in FILE_GROUPS[key][1]:
            for path in glob.glob(pattern):
                try:
                    os.remove(path)
                except OSError as e:
                    print("Could not delete", path, "-", e)
conn.commit()
conn.execute("VACUUM")
conn.close()
print("Done.")