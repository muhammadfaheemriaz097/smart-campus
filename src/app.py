import hashlib
import hmac
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import glob
import time
from db import get_conn, init_db, now_str

st.set_page_config(page_title="Smart Campus Safety", layout="wide")
init_db()
from db import init_db, seed_users
init_db()
seed_users([
    ("admin",   "ChooseStrongPass1", "admin"),
    ("guard",   "ChooseStrongPass2", "security"),
    ("teacher", "ChooseStrongPass3", "teacher"),
])

ROLE_PAGES = {
    "admin": ["Overview", "Live", "Alerts", "Attendance", "Analytics"],
    "security": ["Overview", "Live", "Alerts"],
    "teacher": ["Attendance"],
}


def audit(user, action, detail=""):
    conn = get_conn()
    conn.execute("INSERT INTO audit (time, username, action, detail) VALUES (?,?,?,?)",
                 (now_str(), user, action, detail))
    conn.commit()
    conn.close()


def check_login(username, password):
    conn = get_conn()
    row = conn.execute("SELECT salt, pw_hash, role FROM users WHERE username = ?",
                       (username,)).fetchone()
    conn.close()
    if not row:
        return None
    salt, stored, role = row
    test = hashlib.pbkdf2_hmac("sha256", password.encode(),
                               bytes.fromhex(salt), 200_000).hex()
    return role if hmac.compare_digest(test, stored) else None


def login_view():
    st.title("Smart Campus Safety & Monitoring")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            role = check_login(username.strip(), password)
            if role:
                st.session_state.user = username.strip()
                st.session_state.role = role
                audit(username.strip(), "login")
                st.rerun()
            else:
                audit(username.strip() or "?", "failed_login")
                st.error("Wrong username or password")


@st.fragment(run_every=5)
def overview_page():
    st.subheader("Live overview (refreshes every 5 seconds)")
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_conn()

    def one(sql, params=()):
        return conn.execute(sql, params).fetchone()[0]

    entries = one("SELECT COUNT(*) FROM events WHERE type='ENTRY' AND time LIKE ?", (today + "%",))
    exits = one("SELECT COUNT(*) FROM events WHERE type='EXIT' AND time LIKE ?", (today + "%",))
    open_alerts = one("SELECT COUNT(*) FROM events WHERE review='open' "
                      "AND type IN ('RESTRICTED_ZONE','OVER_CAPACITY')")
    present = one("SELECT COUNT(*) FROM attendance WHERE date = ?", (today,))
    occ = conn.execute("SELECT occupancy, capacity FROM occupancy ORDER BY id DESC LIMIT 1").fetchone()
    recent = pd.read_sql_query(
        "SELECT time, type, detail, review FROM events "
        "WHERE type IN ('RESTRICTED_ZONE','OVER_CAPACITY') ORDER BY id DESC LIMIT 5", conn)
    conn.close()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Entries today", entries)
    c2.metric("Exits today", exits)
    c3.metric("Net inside (today)", entries - exits)
    c4.metric("Open alerts", open_alerts)
    c5.metric("Present today", present)
    if occ:
        st.metric("Current occupancy", f"{occ[0]} / {occ[1]}")
    st.markdown("**Latest alerts**")
    st.dataframe(recent, use_container_width=True, hide_index=True)


def alerts_page():
    st.subheader("Alert review")
    status = st.selectbox("Show", ["open", "real", "false", "all"])
    conn = get_conn()
    sql = ("SELECT id, time, camera, type, track_id, detail, snapshot, review, reviewed_by "
           "FROM events WHERE type IN ('RESTRICTED_ZONE','OVER_CAPACITY')")
    params = ()
    if status != "all":
        sql += " AND review = ?"
        params = (status,)
    df = pd.read_sql_query(sql + " ORDER BY id DESC LIMIT 200", conn, params=params)
    st.dataframe(df.drop(columns=["snapshot"]), use_container_width=True, hide_index=True)

    if not df.empty:
        alert_id = st.selectbox("Open alert ID", df["id"].tolist())
        row = df[df["id"] == alert_id].iloc[0]
        snap = row["snapshot"]
        if snap and os.path.exists(snap):
            st.image(snap, caption=f"{row['type']} at {row['time']}", width=520)
        elif row["type"] == "OVER_CAPACITY":
            st.info("Overcrowding alerts have no snapshot. Details: " + str(row["detail"]))
        b1, b2 = st.columns(2)
        for col, label, value in [(b1, "Mark as REAL", "real"), (b2, "Mark as FALSE ALARM", "false")]:
            if col.button(label):
                conn.execute("UPDATE events SET review=?, reviewed_by=?, reviewed_at=? WHERE id=?",
                             (value, st.session_state.user, now_str(), int(alert_id)))
                conn.commit()
                audit(st.session_state.user, "alert_review", f"alert {alert_id} -> {value}")
                conn.close()
                st.rerun()

    false_n = conn.execute("SELECT COUNT(*) FROM events WHERE review='false'").fetchone()[0]
    real_n = conn.execute("SELECT COUNT(*) FROM events WHERE review='real'").fetchone()[0]
    st.caption(f"Reviewed so far: {real_n} real, {false_n} false alarms")
    conn.close()


def attendance_page():
    st.subheader("Attendance")
    conn = get_conn()
    classes = [r[0] for r in conn.execute("SELECT DISTINCT class_name FROM attendance")] or ["CS-101"]
    c1, c2 = st.columns(2)
    class_name = c1.selectbox("Class", classes)
    date = c2.date_input("Date").strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT name, time FROM attendance WHERE class_name = ? AND date = ? ORDER BY time",
        conn, params=(class_name, date))
    conn.close()
    st.metric("Present", len(df))
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False),
                       file_name=f"attendance_{class_name}_{date}.csv")
    audit(st.session_state.user, "view_attendance", f"{class_name} {date}")


def analytics_page():
    st.subheader("History and trends")
    conn = get_conn()

    ev = pd.read_sql_query("SELECT time, type FROM events WHERE type IN ('ENTRY','EXIT')", conn)
    st.markdown("**Entries and exits per hour**")
    if ev.empty:
        st.info("No entry/exit data yet. Run stage2_count.py first.")
    else:
        ev["hour"] = ev["time"].str[:13]
        st.bar_chart(ev.groupby(["hour", "type"]).size().unstack(fill_value=0))

    occ = pd.read_sql_query(
        "SELECT time, occupancy, capacity FROM occupancy ORDER BY id DESC LIMIT 600", conn)
    st.markdown("**Occupancy vs capacity**")
    if occ.empty:
        st.info("No occupancy data yet. Run stage4_crowd.py first.")
    else:
        occ = occ.iloc[::-1]
        occ["time"] = pd.to_datetime(occ["time"])
        st.line_chart(occ.set_index("time")[["occupancy", "capacity"]])

    al = pd.read_sql_query(
        "SELECT substr(time,1,10) AS day, type FROM events "
        "WHERE type IN ('RESTRICTED_ZONE','OVER_CAPACITY')", conn)
    st.markdown("**Alerts per day**")
    if al.empty:
        st.info("No alerts yet.")
    else:
        st.bar_chart(al.groupby(["day", "type"]).size().unstack(fill_value=0))

    if st.session_state.role == "admin":
        st.markdown("**Audit log (latest 50)**")
        st.dataframe(pd.read_sql_query(
            "SELECT time, username, action, detail FROM audit ORDER BY id DESC LIMIT 50", conn),
            use_container_width=True, hide_index=True)
    conn.close()

@st.fragment(run_every=1)
def live_page():
    st.subheader("Live camera (about 2 frames per second)")
    files = sorted(glob.glob("data/live/*.jpg"))
    if not files:
        st.info("No live feed yet. Start a camera script, for example "
                "stage3_zone.py, then come back here.")
        return
    feeds = [os.path.splitext(os.path.basename(f))[0] for f in files]
    choice = st.selectbox("Feed", feeds, key="live_feed")
    path = f"data/live/{choice}.jpg"
    try:
        age = time.time() - os.path.getmtime(path)
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return
    if age > 5:
        st.warning(f"This feed stopped {int(age)} seconds ago. "
                   "The camera script is probably not running.")
    st.image(data, width=720)

if "user" not in st.session_state:
    login_view()
    st.stop()

pages = ROLE_PAGES[st.session_state.role]
with st.sidebar:
    st.write(f"**{st.session_state.user}** ({st.session_state.role})")
    page = st.radio("Page", pages)
    if st.button("Log out"):
        audit(st.session_state.user, "logout")
        st.session_state.clear()
        st.rerun()

st.title("Smart Campus Safety & Monitoring")
if page == "Overview":
    overview_page()
elif page == "Live":
    live_page()
elif page == "Alerts":
    alerts_page()
elif page == "Attendance":
    attendance_page()
else:
    analytics_page()
