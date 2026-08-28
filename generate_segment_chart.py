# generate_segment_chart.py
import matplotlib.pyplot as plt
import numpy as np

boroughs = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
dispatch_baseline = [102.71, 102.52, 107.37, 109.12, 122.55]
dispatch_model = [98.74, 98.92, 105.50, 106.33, 120.13]

x = np.arange(len(boroughs))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width/2, dispatch_baseline, width, label="Baseline", color="#888")
ax.bar(x + width/2, dispatch_model, width, label="Model", color="#2ca02c")
ax.set_xticks(x); ax.set_xticklabels(boroughs)
ax.set_ylabel("Dispatch MAE (seconds)")
ax.set_title("Dispatch: model beats baseline in every borough")
ax.legend()
plt.tight_layout()
plt.savefig("dispatch_by_borough.png", dpi=150)

trip_boroughs = ["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island"]
trip_baseline = [392.80, 359.91, 2249.27, 446.31, 541.20, 492.27]
trip_model = [181.28, 185.10, 434.55, 213.79, 210.72, 184.22]

x = np.arange(len(trip_boroughs))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width/2, trip_baseline, width, label="Baseline", color="#888")
ax.bar(x + width/2, trip_model, width, label="Model", color="#2ca02c")
ax.set_xticks(x); ax.set_xticklabels(trip_boroughs)
ax.set_ylabel("Trip MAE (seconds)")
ax.set_title("Trip: biggest win on EWR — baseline badly misjudges airport routes")
ax.legend()
plt.tight_layout()
plt.savefig("trip_by_borough.png", dpi=150)
print("saved: dispatch_by_borough.png, trip_by_borough.png")