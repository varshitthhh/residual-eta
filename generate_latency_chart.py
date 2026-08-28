# generate_latency_chart.py
import matplotlib.pyplot as plt

configs = ["Buggy\n(1 worker)", "Fixed calls\n(1 worker)", "Fixed calls\n(4 workers)"]
throughput = [7.7, 14.2, 85.7]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(configs, throughput, color=["#d62728", "#ff7f0e", "#2ca02c"])
ax.set_ylabel("Throughput (req/sec, 20 concurrent)")
ax.set_title("Fixing redundant model calls + multi-worker deployment")
for bar, val in zip(bars, throughput):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val}", ha="center")
plt.tight_layout()
plt.savefig("throughput_comparison.png", dpi=150)

concurrency = [5, 10, 20, 40, 80]
req_sec = [24.2, 47.4, 57.5, 77.0, 79.2]
p95_ms = [345.3, 608.0, 462.7, 646.3, 1042.3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(concurrency, req_sec, marker="o", color="#2ca02c")
ax1.set_xlabel("Concurrent requests"); ax1.set_ylabel("Throughput (req/sec)")
ax1.set_title("4-worker throughput saturates ~78-79 req/sec")
ax2.plot(concurrency, p95_ms, marker="o", color="#d62728")
ax2.set_xlabel("Concurrent requests"); ax2.set_ylabel("p95 latency (ms)")
ax2.set_title("Latency degrades gracefully under load")
plt.tight_layout()
plt.savefig("concurrency_sweep.png", dpi=150)
print("saved: throughput_comparison.png, concurrency_sweep.png")