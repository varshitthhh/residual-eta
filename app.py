"""Two-leg rideshare ETA API — serves 4 LightGBM models trained on NYC HVFHS data."""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import lightgbm as lgb
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOOKUPS_DIR = BASE_DIR / "lookups"

app = FastAPI(title="Rideshare ETA API")

# ---------- load everything once at startup ----------

dispatch_models = {
    "point": lgb.Booster(model_file=str(MODELS_DIR / "dispatch_point.txt")),
    "safe": lgb.Booster(model_file=str(MODELS_DIR / "dispatch_safe.txt")),
}
trip_models = {
    "point": lgb.Booster(model_file=str(MODELS_DIR / "trip_point.txt")),
    "safe": lgb.Booster(model_file=str(MODELS_DIR / "trip_safe.txt")),
}

with open(MODELS_DIR / "calibration.json") as f:
    calibration = json.load(f)

# fix: PULocationID/DOLocationID were trained as integer categories but got
# stringified during the Colab export for JSON safety. Coerce them back here
# rather than silently letting LightGBM treat every zone as unseen.
INTEGER_CATEGORICAL_COLS = {"PULocationID", "DOLocationID"}
with open(MODELS_DIR / "category_levels.json") as f:
    _raw_category_levels = json.load(f)
category_levels = {
    col: ([int(x) for x in levels] if col in INTEGER_CATEGORICAL_COLS else levels)
    for col, levels in _raw_category_levels.items()
}

zones = pd.read_csv(LOOKUPS_DIR / "zone_centroids.csv")
zone_lookup = zones.set_index("location_id").to_dict("index")

dispatch_zone_hour = pd.read_csv(LOOKUPS_DIR / "dispatch_zone_hour.csv")
dispatch_borough_hour = pd.read_csv(LOOKUPS_DIR / "dispatch_borough_hour.csv")
speed_by_hour = pd.read_csv(LOOKUPS_DIR / "speed_by_hour.csv").set_index("hour")["avg_mph"].to_dict()
toll_route_pct = pd.read_csv(LOOKUPS_DIR / "toll_route_pct.csv")
toll_borough_pct = pd.read_csv(LOOKUPS_DIR / "toll_borough_pct.csv")

recent_demand = pd.read_csv(LOOKUPS_DIR / "recent_zone_demand_snapshot.csv").set_index("location_id")["recent_zone_demand_snapshot"].to_dict()
recent_speed = pd.read_csv(LOOKUPS_DIR / "recent_zone_speed_snapshot.csv").set_index("location_id")["recent_zone_speed_snapshot"].to_dict()
# fallback for a zone missing from the snapshot: citywide average of what we do have.
# Demo simplification -- the original training pipeline filled gaps with a
# TRAIN-period zone-level average that lives only inside the Colab warehouse
# and wasn't exported. This is a reasonable stand-in, not the identical fallback.
fallback_demand = float(np.mean(list(recent_demand.values())))
fallback_speed = float(np.mean(list(recent_speed.values())))

DISPATCH_FEATURES = ["PULocationID", "pu_borough", "company", "dispatch_weather_sev",
                      "dispatch_hour", "day_of_week", "recent_zone_demand_filled"]
TRIP_FEATURES = ["PULocationID", "DOLocationID", "do_borough", "trip_weather_sev",
                  "trip_hour", "day_of_week", "recent_zone_speed_filled",
                  "trip_miles", "toll_route_pct_filled", "shared_request_flag"]
CATEGORICAL_COLS = ["PULocationID", "DOLocationID", "pu_borough", "do_borough", "company",
                     "dispatch_weather_sev", "trip_weather_sev", "shared_request_flag"]


class ETARequest(BaseModel):
    pickup_zone: int
    dropoff_zone: int
    request_datetime: str            # ISO format, e.g. "2026-08-07T14:30:00"
    company: str                     # e.g. "HV0003"
    weather: str = "dry"             # dry | light_rain | heavy_rain | snow
    shared_request: bool = False
    trip_miles_override: Optional[float] = None   # pass real distance if you know it


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def duckdb_day_of_week(dt: datetime) -> int:
    # DuckDB DAYOFWEEK: Sunday=0 ... Saturday=6. Python .weekday(): Monday=0 ... Sunday=6.
    # Verified against a known reference date -- getting this backwards silently
    # trains/serves on the wrong day, the same failure class as the earlier
    # request-vs-pickup timestamp bug.
    return (dt.weekday() + 1) % 7


def get_dispatch_baseline(pu_zone: int, hour: int) -> float:
    row = dispatch_zone_hour[(dispatch_zone_hour.location_id == pu_zone) & (dispatch_zone_hour.hour == hour)]
    if len(row) and row.iloc[0]["n"] >= 50:
        return float(row.iloc[0]["zone_hour_avg_dispatch"])
    borough = zone_lookup[pu_zone]["borough"]
    fb = dispatch_borough_hour[(dispatch_borough_hour.borough == borough) & (dispatch_borough_hour.hour == hour)]
    if len(fb):
        return float(fb.iloc[0]["borough_hour_avg_dispatch"])
    # third-tier fallback: citywide average at this hour, across all boroughs.
    # Needed for genuinely thin zones like EWR -- almost no rideshare pickups
    # originate there in this dataset, since it's overwhelmingly a drop-off
    # destination, so even the borough level has no history to fall back on.
    citywide = dispatch_borough_hour[dispatch_borough_hour.hour == hour]["borough_hour_avg_dispatch"]
    if len(citywide):
        return float(citywide.mean())
    raise HTTPException(400, f"No dispatch data available anywhere for hour {hour}")


def get_trip_baseline(trip_miles: float, hour: int) -> float:
    avg_mph = speed_by_hour.get(hour)
    if avg_mph is None:
        raise HTTPException(400, f"No speed baseline for hour {hour}")
    return trip_miles / avg_mph * 3600


def get_toll_pct(pu_zone: int, do_zone: int, pu_borough: str, do_borough: str) -> float:
    row = toll_route_pct[(toll_route_pct.pu == pu_zone) & (toll_route_pct.do_id == do_zone)]
    if len(row) and row.iloc[0]["n"] >= 20:
        return float(row.iloc[0]["toll_route_pct"])
    fb = toll_borough_pct[(toll_borough_pct.pu_borough == pu_borough) & (toll_borough_pct.do_borough == do_borough)]
    if len(fb):
        return float(fb.iloc[0]["toll_borough_pct"])
    return 0.0  # no history at all -- assume no toll, stated explicitly, not hidden


def build_row(values: dict, feature_list: list) -> pd.DataFrame:
    row = {}
    for col in feature_list:
        if col in CATEGORICAL_COLS:
            val = int(values[col]) if col in INTEGER_CATEGORICAL_COLS else str(values[col])
            if val not in category_levels[col]:
                raise HTTPException(400, f"Unknown value '{val}' for '{col}' -- not seen in training data")
            row[col] = pd.Categorical([val], categories=category_levels[col])
        else:
            row[col] = [values[col]]
    return pd.DataFrame(row)

def predict_with_explanation(model, row, feature_list):
    contrib = model.predict(row[feature_list], pred_contrib=True)[0]
    feature_contribs = contrib[:-1]
    bias = float(contrib[-1])
    point_prediction = float(contrib.sum())   # bias + all contributions = the plain prediction, for free
    sorted_contribs = sorted(
        [{"feature": f, "contribution_seconds": round(float(v), 1)} for f, v in zip(feature_list, feature_contribs)],
        key=lambda x: -abs(x["contribution_seconds"]),
    )
    return point_prediction, {"base_value_seconds": round(bias, 1), "feature_contributions": sorted_contribs}

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/zones")
def get_zones():
    return zones[["location_id", "zone", "borough"]].to_dict("records")


@app.get("/categories")
def get_categories():
    return {
        "company": category_levels["company"],
        "valid_pickup_zones": category_levels["PULocationID"],
        "valid_dropoff_zones": category_levels["DOLocationID"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: ETARequest):
    if req.pickup_zone not in zone_lookup:
        raise HTTPException(400, f"Unknown pickup_zone {req.pickup_zone}")
    if req.dropoff_zone not in zone_lookup:
        raise HTTPException(400, f"Unknown dropoff_zone {req.dropoff_zone}")

    dt = datetime.fromisoformat(req.request_datetime)
    dispatch_hour = dt.hour
    trip_hour = dt.hour
    day_of_week = duckdb_day_of_week(dt)

    pu_borough = zone_lookup[req.pickup_zone]["borough"]
    do_borough = zone_lookup[req.dropoff_zone]["borough"]

    if req.trip_miles_override is not None:
        trip_miles = req.trip_miles_override
    else:
        pu, do = zone_lookup[req.pickup_zone], zone_lookup[req.dropoff_zone]
        trip_miles = haversine_miles(pu["lat"], pu["lon"], do["lat"], do["lon"])

    recent_demand_val = recent_demand.get(req.pickup_zone, fallback_demand)
    recent_speed_val = recent_speed.get(req.pickup_zone, fallback_speed)
    toll_pct = get_toll_pct(req.pickup_zone, req.dropoff_zone, pu_borough, do_borough)

    dispatch_baseline = get_dispatch_baseline(req.pickup_zone, dispatch_hour)
    trip_baseline = get_trip_baseline(trip_miles, trip_hour)

    dispatch_row = build_row({
        "PULocationID": req.pickup_zone, "pu_borough": pu_borough, "company": req.company,
        "dispatch_weather_sev": req.weather, "dispatch_hour": dispatch_hour,
        "day_of_week": day_of_week, "recent_zone_demand_filled": recent_demand_val,
    }, DISPATCH_FEATURES)

    trip_row = build_row({
        "PULocationID": req.pickup_zone, "DOLocationID": req.dropoff_zone, "do_borough": do_borough,
        "trip_weather_sev": req.weather, "trip_hour": trip_hour, "day_of_week": day_of_week,
        "recent_zone_speed_filled": recent_speed_val, "trip_miles": trip_miles,
        "toll_route_pct_filled": toll_pct, "shared_request_flag": "Y" if req.shared_request else "N",
    }, TRIP_FEATURES)

    # single pred_contrib call per point model does double duty: the prediction
    # AND the explanation, instead of the earlier 6-call version that computed
    # the point prediction twice (once plain, once again inside a separate explain() call)
    dispatch_residual_p50, dispatch_explanation = predict_with_explanation(dispatch_models["point"], dispatch_row, DISPATCH_FEATURES)
    dispatch_p50 = dispatch_baseline + dispatch_residual_p50
    dispatch_p90 = dispatch_baseline + dispatch_models["safe"].predict(dispatch_row[DISPATCH_FEATURES])[0] + calibration["dispatch_shift"]

    trip_residual_p50, trip_explanation = predict_with_explanation(trip_models["point"], trip_row, TRIP_FEATURES)
    trip_p50 = trip_baseline + trip_residual_p50
    trip_p90 = trip_baseline + trip_models["safe"].predict(trip_row[TRIP_FEATURES])[0] + calibration["trip_shift"]

    dispatch_p90 = max(dispatch_p90, dispatch_p50)
    trip_p90 = max(trip_p90, trip_p50)

    return {
        "dispatch_eta_seconds": {"p50": round(dispatch_p50, 1), "p90": round(dispatch_p90, 1)},
        "trip_eta_seconds": {"p50": round(trip_p50, 1), "p90": round(trip_p90, 1)},
        "total_eta_seconds": {
            "p50": round(dispatch_p50 + trip_p50, 1),
            "p90_approx": round(dispatch_p90 + trip_p90, 1),
        },
        "trip_miles_used": round(trip_miles, 2),
        "dispatch_explanation": dispatch_explanation,
        "trip_explanation": trip_explanation,
    }


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")