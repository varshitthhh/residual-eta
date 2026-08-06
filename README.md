**Save this as `README.md` in your project root** (`C:\Users\sriva\Downloads\rideshare-eta\README.md`):

```markdown
# Two-Leg Rideshare ETA — Residual-Corrected, Calibrated Prediction System

Predicts ride-hailing arrival times as **two separate predictions** — how long until
the driver arrives (dispatch), and how long the ride itself takes (trip) — instead of
one end-to-end guess. Trained on 55M real NYC rideshare trips, deployed as a working
prediction API with explainability and measured latency/throughput.

---

## In plain terms

When you book a ride, the app shows you two numbers at two different moments:
"driver arriving in 4 min," then later "trip will take 18 min." These are genuinely
different problems — dispatch time depends on how many drivers are nearby, trip time
depends on distance and traffic — so this project builds them as two separate models
instead of forcing one model to learn both patterns at once.

Instead of predicting the ETA from scratch, each model predicts the **error** in a
simple baseline guess (a historical average, or distance ÷ average speed) and corrects
it — the same design Uber's production ETA system (DeepETA) uses. Rather than a single
number, each prediction comes with a "likely" estimate and a calibrated "safe, worst-
realistic-case" estimate, so the uncertainty is honest instead of hidden.

---

## Why this isn't a standard ETA notebook

- **Two-leg decomposition** with per-leg feature availability — dispatch-leg features
  are computed from what's knowable at *request* time, trip-leg features from what's
  knowable at *pickup* time. Mixing these up is a real leakage bug this project found
  and fixed (see below).
- **Residual correction**, not raw prediction — each model learns the *gap* from a
  baseline, not the ETA itself.
- **Per-leg asymmetric loss chosen by evidence, not assumed.** Quantile regression's
  alpha parameter *is* the business cost ratio between early/late errors. Testing
  showed alpha=0.65 helps the trip leg but actively hurts the dispatch leg (which has
  a much stronger baseline with less residual signal to redirect) — so dispatch uses
  alpha=0.5, trip uses alpha=0.65. Same technique, different verdict per leg, both
  backed by bootstrap-confidence-interval evidence.
- **P90 estimates are calibrated, not just labeled.** Raw quantile regression from
  GBDTs doesn't reliably hit its nominal coverage (measured: 85% actual vs. 90% target)
  — fixed with a validation-set calibration shift, verified to hold on a true holdout.
- **Explainability that actually reconciles.** Uses LightGBM's native `pred_contrib`
  (exact TreeSHAP, no external dependency) — verified by hand that
  `base_value + feature_contributions = prediction`, not just plotted and trusted.
- **Measured, not assumed, latency/throughput.** Found and fixed a real performance
  bug (duplicate model calls per request cost 2x throughput), then benchmarked
  single-worker vs. multi-worker deployment with a real concurrency sweep.

---

## Data

[NYC TLC High-Volume For-Hire Services (HVFHS)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
trip records — real, publicly released Uber/Lyft/Via/Juno trip data.

- **Period:** February, June, October 2024 (seasonal spread, ~54.7M rows after cleaning
  from 59.5M raw — 91.9% retention)
- **Split:** time-based, not random — Feb → train, Jun → validation, Oct → test
- **Key fields used:** `request_datetime`, `pickup_datetime`, `dropoff_datetime`,
  pickup/dropoff zone, `trip_miles`, tolls, shared-ride flag

**Known data limitations, stated explicitly rather than hidden:**
- No raw GPS — TLC anonymizes to 263 pickup/dropoff zones. Distance features use
  zone-centroid straight-line estimates, not real road-network distance.
- No driver starting location — the dispatch leg has no distance signal at all,
  which is why its baseline is a historical statistical average, not a physics model.
- "Recent zone activity" features approximate real-time signal using 15-minute
  historical buckets, since no live GPS feed exists in this public dataset.

---

## Architecture / workflow

```
Raw HVFHS Parquet (DuckDB, out-of-core)
        │
        ▼
Cleaning + zone-centroid join + weather join   (percentile-capped, leakage-checked)
        │
        ▼
Time-based split (Feb / Jun / Oct)
        │
        ▼
Baselines: dispatch = zone×hour historical avg (borough → citywide fallback)
           trip     = trip_miles ÷ avg speed-by-hour
        │
        ▼
Feature engineering (effect-size validated, not assumed) + residual targets
        │
        ▼
LightGBM × 4 (dispatch/trip × point/safe quantile models)
        │
        ▼
P90 calibration (validation-set shift, verified on true test holdout)
        │
        ▼
Evaluation: MAE vs. baseline, bootstrap CI, P90 coverage, segment breakdown
        │
        ▼
FastAPI + HTML deployment, with SHAP-equivalent explanation per prediction
```

---

## Results (test set, true holdout)

| Leg | Baseline MAE | Model MAE | Improvement | 95% CI |
|---|---|---|---|---|
| Dispatch | 106.1s | 103.3s | +2.79s (2.6%) | [2.65, 2.93] |
| Trip | 450.1s | 202.6s | +247.5s (55%) | [245.2, 249.8] |

P90 empirical coverage on test: dispatch 91.4%, trip 89.5% (target: 90%).

**Segment finding:** the model's biggest win is on structurally atypical routes —
airport (EWR) trips saw baseline MAE of 2249s cut to 434s (81% reduction), since a
flat hourly-average-speed baseline badly misjudges highway-heavy routes.

**Latency/throughput** (local benchmark, 20-core machine):
- Unloaded: p50 ≈ 52ms, p99 ≈ 90ms
- Single-worker capacity ceiling: ~23-25 req/sec (flat regardless of concurrency)
- 4-worker capacity ceiling: ~78-79 req/sec (~3.3x scaling — sub-linear due to shared
  resource overhead, not a bug)

---

## Requirements

```
fastapi
uvicorn[standard]
lightgbm
pandas
numpy
```

Python 3.9+ (uses `typing.Optional`, not the `|` union syntax, for broader compatibility).

## Project structure

```
rideshare-eta/
├── models/              # trained LightGBM models + calibration + category encodings
├── lookups/              # baseline & feature lookup tables (zone×hour, speed, tolls)
├── static/
│   └── index.html        # simple web UI
├── app.py                 # FastAPI backend
├── load_test.py            # latency/throughput benchmark script
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
uvicorn app:app --workers 4
```

Open `http://127.0.0.1:8000` in a browser. For development (auto-reload on code
changes, but not representative of real performance), use `uvicorn app:app --reload`
instead — single-worker only.

## API

- `GET /zones` — list of valid pickup/dropoff zones
- `GET /categories` — valid company codes, and which zones are valid as pickup vs.
  dropoff (some zones, like EWR, have essentially no pickup history in this dataset)
- `POST /predict` — takes pickup/dropoff zone, request time, company, weather,
  shared-ride flag, optional real trip distance; returns P50/P90 for both legs plus
  a per-feature explanation of each prediction
- `GET /health` — basic liveness check

## Limitations

- Trip distance defaults to a straight-line (haversine) estimate between zone
  centroids unless a real value is passed in `trip_miles_override` — there's no
  routing engine in this project by design (no raw GPS available in the source data).
- "Recent zone activity" features use a frozen historical snapshot, not a live feed —
  a real production version would stream this from active trip data.
- The combined "total ETA" P90 is an approximate sum of both legs' P90s, not the exact
  P90 of their joint distribution (that would require modeling their correlation).
- Hyperparameters were set by evidence-driven manual tuning, not a full Optuna search —
  scoped down given local compute constraints (2-core Colab sessions during training).
```