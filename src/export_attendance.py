import sqlite3
from datetime import datetime

import pandas as pd

from secure_store import load_faces

CLASS_NAME = "CS-101"      # must match CLASS_NAME in stage5_attendance.py

date = input("Date (YYYY-MM-DD, press Enter for today): ").strip()
date = date or datetime.now().strftime("%Y-%m-%d")

enrolled = list(load_faces())
if not enrolled:
    raise SystemExit("No enrolled people found. Run enroll.py first.")

conn = sqlite3.connect("data/campus.db")
marked = pd.read_sql_query(
    "SELECT name, time FROM attendance WHERE class_name = ? AND date = ?",
    conn, params=(CLASS_NAME, date))
conn.close()

report = pd.DataFrame({"name": enrolled}).merge(marked, on="name", how="left")
report["status"] = report["time"].notna().map({True: "Present", False: "Absent"})
report["time"] = report["time"].fillna("")
report.insert(0, "class", CLASS_NAME)
report.insert(0, "date", date)

out_file = f"data/attendance_{CLASS_NAME}_{date}.csv"
report.to_csv(out_file, index=False)

print(report.to_string(index=False))
print(f"\nPresent: {(report['status'] == 'Present').sum()} / {len(report)}")
print(f"Saved to {out_file}")
