# load_test.py
import time
import statistics
import concurrent.futures
import requests

BASE = "http://127.0.0.1:8000"

categories = requests.get(f"{BASE}/categories").json()
PAYLOAD = {
    "pickup_zone": categories["valid_pickup_zones"][0],
    "dropoff_zone": categories["valid_dropoff_zones"][0],
    "request_datetime": "2026-08-07T14:30:00",
    "company": categories["company"][0],
    "weather": "dry",
    "shared_request": False,
}

def single_request():
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/predict", json=PAYLOAD)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return elapsed

single_request()  # warm-up, excluded from measurement -- first call includes cold-start effects

latencies_ms = sorted(single_request() * 1000 for _ in range(50))
p50, p95, p99 = latencies_ms[25], latencies_ms[47], latencies_ms[49]
print(f"Sequential (n=50): p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  mean={statistics.mean(latencies_ms):.1f}ms")

N_CONCURRENT, N_TOTAL = 20, 200
t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
    results = list(ex.map(lambda _: single_request(), range(N_TOTAL)))
total_time = time.perf_counter() - t0
concurrent_ms = sorted(r * 1000 for r in results)
print(f"Concurrent ({N_CONCURRENT} at a time, n={N_TOTAL}): {N_TOTAL/total_time:.1f} req/sec")
print(f"  under load: p50={concurrent_ms[100]:.1f}ms  p95={concurrent_ms[189]:.1f}ms")

# add to load_test.py, after the existing concurrent test
# fixed sample count per concurrency level, not scaled with concurrency --
# needed for percentiles that are actually stable enough to compare
for n_concurrent in [5, 10, 20, 40, 80]:
    N_SAMPLES = 100
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        results = list(ex.map(lambda _: single_request(), range(N_SAMPLES)))
    total_time = time.perf_counter() - t0
    ms = sorted(r * 1000 for r in results)
    print(f"concurrency={n_concurrent}: {len(results)/total_time:.1f} req/sec, p50={ms[49]:.1f}ms, p95={ms[94]:.1f}ms")