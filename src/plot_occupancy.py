import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/occupancy_log.csv", parse_dates=["time"])
if df.empty:
    raise SystemExit("occupancy_log.csv has no rows yet. Run stage4_crowd.py first.")

capacity = df["capacity"].iloc[-1]

plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["occupancy"], label="Occupancy")
plt.axhline(capacity, color="red", linestyle="--", label=f"Capacity ({capacity})")
plt.xlabel("Time")
plt.ylabel("People in area")
plt.title("Crowd occupancy over time")
plt.legend()
plt.tight_layout()
plt.savefig("data/occupancy_chart.png", dpi=150)
plt.show()

print("Peak occupancy:", df["occupancy"].max())
print("Average occupancy:", round(df["occupancy"].mean(), 2))
print("Seconds over capacity:", int((df["occupancy"] > df["capacity"]).sum()))

alerts = pd.read_csv("data/crowd_alerts.csv")
print("Over-capacity alerts:", int((alerts["event"] == "OVER_CAPACITY").sum()))