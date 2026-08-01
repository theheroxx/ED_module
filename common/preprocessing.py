"""
common/preprocessing.py
=======================
Shared weather preprocessing used by EVERY step. Living in `common/` (not in
step1) is what keeps the steps decoupled: they share PREPROCESSING, never each
other's analytical output.

Key entry point:
    get_processed_data(raw_path=None, use_cache=True) -> pd.DataFrame

It loads the raw CSV, cleans it, attaches the ED baseline columns (via
common.ed_baseline), derives discretized categorical bins for association
mining, adds a latitude-based climate zone, and caches the result to
config.PROCESSED_DATA_PATH so downstream steps reuse it without recomputing.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from common.ed_baseline import compute_ed_baseline_frame


# ----------------------------------------------------------------------------
# Discretizers for association-rule "transaction" items
# ----------------------------------------------------------------------------
def _bin_temperature(t):
    conds = [t < 0, t < 10, t < 18, t < 25, t < 30, t < 35]
    labels = ["TEMP_FREEZING", "TEMP_COLD", "TEMP_COOL", "TEMP_IDEAL",
              "TEMP_WARM", "TEMP_HOT"]
    return np.select(conds, labels, default="TEMP_VERY_HOT")


def _bin_aqi(epa):
    m = {1: "AQI_GOOD", 2: "AQI_MODERATE", 3: "AQI_SENSITIVE",
         4: "AQI_UNHEALTHY", 5: "AQI_VERY_UNHEALTHY", 6: "AQI_HAZARDOUS"}
    return pd.Series(epa).map(m).fillna("AQI_UNKNOWN").to_numpy()


def _bin_humidity(h):
    conds = [h < 30, h < 60, h < 80]
    labels = ["HUM_DRY", "HUM_COMFORTABLE", "HUM_HUMID"]
    return np.select(conds, labels, default="HUM_VERY_HUMID")


def _bin_wind(w):
    conds = [w < 10, w < 20, w < 40]
    labels = ["WIND_CALM", "WIND_BREEZY", "WIND_WINDY"]
    return np.select(conds, labels, default="WIND_STRONG")


def _bin_uv(u):
    conds = [u < 3, u < 6, u < 8, u < 11]
    labels = ["UV_LOW", "UV_MODERATE", "UV_HIGH", "UV_VERY_HIGH"]
    return np.select(conds, labels, default="UV_EXTREME")


def _climate_zone(lat, temp, hum):
    """Coarse Koppen-inspired zone from latitude/temperature/humidity."""
    lat = np.abs(np.asarray(lat, dtype=float))
    zones = np.select(
        [lat < 23.5, lat < 35, lat < 66.5],
        ["ZONE_TROPICAL", "ZONE_SUBTROPICAL", "ZONE_TEMPERATE"],
        default="ZONE_POLAR",
    )
    return zones


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
DISCRETE_COLS = ["temp_bin", "aqi_bin", "hum_bin", "wind_bin", "uv_bin",
                 "climate_zone", "ed_category"]


def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Numeric coercion for the fields we rely on
    numeric_cols = [
        "temperature_celsius", "temperature_fahrenheit",
        "humidity", "wind_kph", "wind_mph", "wind_degree",
        "uv_index", "feels_like_celsius", "feels_like_fahrenheit",
        "air_quality_us-epa-index",
        "air_quality_PM2.5", "air_quality_PM10",
        "air_quality_Ozone", "air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide", "air_quality_Carbon_Monoxide",
        "air_quality_gb-defra-index",
        "latitude", "longitude", "pressure_mb", "pressure_in",
        "precip_mm", "precip_in", "cloud", "visibility_km", "visibility_miles",
        "gust_kph", "gust_mph", "moon_illumination",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ED baseline (the shared source of truth) ------------------------------
    ed = compute_ed_baseline_frame(df)
    df = pd.concat([df, ed], axis=1)

    # Discretized items for association mining ------------------------------
    temp = df.get("temperature_celsius", pd.Series(20.0, index=df.index))
    hum = df.get("humidity", pd.Series(50.0, index=df.index))
    wind = df.get("wind_kph", pd.Series(0.0, index=df.index))
    uv = df.get("uv_index", pd.Series(0.0, index=df.index))
    epa = df.get("air_quality_us-epa-index", pd.Series(np.nan, index=df.index))
    lat = df.get("latitude", pd.Series(0.0, index=df.index))

    df["temp_bin"] = _bin_temperature(temp.to_numpy(dtype=float))
    df["aqi_bin"] = _bin_aqi(epa.to_numpy(dtype=float))
    df["hum_bin"] = _bin_humidity(hum.to_numpy(dtype=float))
    df["wind_bin"] = _bin_wind(wind.to_numpy(dtype=float))
    df["uv_bin"] = _bin_uv(uv.to_numpy(dtype=float))
    df["climate_zone"] = _climate_zone(lat.to_numpy(dtype=float),
                                       temp.to_numpy(dtype=float),
                                       hum.to_numpy(dtype=float))
    return df


def get_processed_data(raw_path=None, use_cache=True) -> pd.DataFrame:
    """Load -> clean -> enrich, with parquet caching. The single shared handoff."""
    raw_path = Path(raw_path) if raw_path else config.RAW_DATA_PATH
    cache = config.PROCESSED_DATA_PATH

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. "
            f"Place GlobalWeatherRepository.csv in {config.DATA_DIR}."
        )

    df = pd.read_csv(raw_path)
    df = clean_and_enrich(df)

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache, index=False)
    except Exception:
        df.to_csv(cache.with_suffix(".csv"), index=False)
    return df


if __name__ == "__main__":
    d = get_processed_data()
    print("Processed:", d.shape)
    print(d[["ed_score", "ed_category"]].describe(include="all"))