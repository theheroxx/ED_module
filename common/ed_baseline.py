"""
common/ed_baseline.py
=====================
SINGLE SOURCE OF TRUTH for the Exercise Danger (ED) baseline score [0-100].

UPDATED (2026-07-24 - FIXED):
  - Fixed NumPy 2.x int() crash in single-row branch
  - _points_air now always returns numpy array (consistent type)
  - Safety floor redesigned: actually pulls score up when any component exceeds threshold
  - Synergy comment corrected (cap is 23, not 21)
  - max_boost parameter removed (dead code)
  - NaN filtering unified across branches
  - Risk categories are the single source of truth (used by math_model.py)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Import pollution module
from common.pollution import calculate_air_quality_points


# ----------------------------------------------------------------------------
# Physiological indices 
# ----------------------------------------------------------------------------
def heat_index_celsius(temp_c, rh):
    temp_c = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh, dtype=float)
    t_f = temp_c * 9.0 / 5.0 + 32.0
    hi_simple = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
    hi_full = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh
               - 0.22475541 * t_f * rh - 0.00683783 * t_f ** 2
               - 0.05481717 * rh ** 2 + 0.00122874 * t_f ** 2 * rh
               + 0.00085282 * t_f * rh ** 2 - 0.00000199 * t_f ** 2 * rh ** 2)
    _sqrt_arg = np.clip((17 - np.abs(t_f - 95.0)) / 17.0, 0.0, None)
    adj_low_rh = np.where((rh < 13) & (t_f >= 80) & (t_f <= 112),
                          ((13 - rh) / 4.0) * np.sqrt(_sqrt_arg), 0.0)
    adj_high_rh = np.where((rh > 85) & (t_f >= 80) & (t_f <= 87),
                           ((rh - 85) / 10.0) * ((87 - t_f) / 5.0), 0.0)
    hi_full = hi_full - adj_low_rh + adj_high_rh
    hi_f = np.where((t_f + hi_simple) / 2.0 < 80.0, hi_simple, hi_full)
    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return np.maximum(hi_c, temp_c)


def wind_chill_celsius(temp_c, wind_kph):
    temp_c = np.asarray(temp_c, dtype=float)
    wind_kph = np.asarray(wind_kph, dtype=float)
    v = np.power(np.clip(wind_kph, 0, None), 0.16)
    wc = 13.12 + 0.6215 * temp_c - 11.37 * v + 0.3965 * temp_c * v
    applies = (temp_c <= 10.0) & (wind_kph > 4.8)
    return np.where(applies, wc, temp_c)


# ----------------------------------------------------------------------------
# Component point functions 
# ----------------------------------------------------------------------------
def _points_heat(apparent_temp_c):
    xp = [27, 32, 41, 54]
    fp = [0, 15, 30, 45]
    return np.interp(apparent_temp_c, xp, fp)


def _points_cold(wind_chill_c):
    xp = [-40, -28, -10, 0]
    fp = [40, 30, 15, 0]
    return np.interp(wind_chill_c, xp, fp)


def _points_air(epa_index, pollutants=None):
    """
    Convert EPA Index + pollutant data → continuous air points (0-100).
    Always returns a numpy array.
    """
    epa_index = np.asarray(epa_index, dtype=float)
    pollutants = pollutants or {}
    
    # Ensure pollutants values are arrays
    for k, v in pollutants.items():
        if v is not None and not isinstance(v, np.ndarray):
            pollutants[k] = np.atleast_1d(v)
    
    # Scalar / single-row branch
    if np.isscalar(epa_index) or len(epa_index) == 1:
        epa_scalar = float(epa_index.item()) if not np.isscalar(epa_index) else float(epa_index)
        poll_dict = {}
        for k, v in pollutants.items():
            if v is not None:
                val = v[0] if isinstance(v, np.ndarray) and len(v) > 0 else v
                if val is not None and not np.isnan(val):
                    poll_dict[k] = val
        return np.array([calculate_air_quality_points(poll_dict, int(epa_scalar))])
    
    # Multi-row branch
    results = []
    for i, epa in enumerate(epa_index):
        if np.isnan(epa):
            results.append(5.0)
            continue
        poll_dict = {}
        for k, v in pollutants.items():
            if v is not None and i < len(v):
                val = v[i]
                if not np.isnan(val):
                    poll_dict[k] = val
        results.append(calculate_air_quality_points(poll_dict, int(epa)))
    
    return np.array(results)


def _points_uv(uv_index):
    xp = [2, 3, 6, 8, 11]
    fp = [0, 3, 6, 9, 12]
    return np.interp(uv_index, xp, fp)


def _score_to_category(score):
    """Single source of truth for risk categories."""
    conds = [score >= 75, score >= 55, score >= 35, score >= 15]
    labels = ["ED_VERY_DANGEROUS", "ED_DANGEROUS", "ED_CAUTION", "ED_MODERATE_SAFE"]
    return np.select(conds, labels, default="ED_VERY_SAFE")


# ============================================================================
# SAFETY FLOOR 
# ============================================================================
def _apply_safety_floor(score, components, threshold=70, margin=5):
    """
    Apply safety floor override if any component is dangerously high.
    
    If any component exceeds `threshold`, the final score is raised to
    at least (max_component + margin), but never above 100.
    
    This fixes the previous design where the sum of non-negative components
    was always ≥ max_component, making the pull-toward logic inert.
    
    Args:
        score: Current ED score (0-100)
        components: dict of component scores
        threshold: Activation threshold (default 70)
        margin: Extra points to add beyond max_component (default 5)
    
    Returns:
        Adjusted ED score (0-100)
    """
    if not components:
        return score
    
    max_component = max(components.values())
    
    if max_component > threshold:
        # Set floor to max_component + margin
        floor = min(100, max_component + margin)
        return max(score, floor)
    
    return score


# ----------------------------------------------------------------------------
# Column helpers
# ----------------------------------------------------------------------------
def _col(df, name, default):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.full(len(df), default), index=df.index, dtype=float)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
COMPONENT_COLS = ["ed_f_heat", "ed_f_cold", "ed_f_air", "ed_f_uv", "ed_f_synergy"]


def compute_ed_baseline_frame(df: pd.DataFrame) -> pd.DataFrame:
    temp = _col(df, "temperature_celsius", 20.0)
    rh = _col(df, "humidity", 50.0).clip(0, 100)
    wind = _col(df, "wind_kph", 0.0).clip(lower=0)
    uv = _col(df, "uv_index", 0.0).clip(lower=0)
    epa = _col(df, "air_quality_us-epa-index", 1.0)

    # ---- COLLECT ALL POLLUTANTS ----
    pollutant_cols = [
        "air_quality_PM2.5",
        "air_quality_PM10",
        "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide",
        "air_quality_Carbon_Monoxide",
    ]
    pollutants = {}
    for col in pollutant_cols:
        if col in df.columns:
            key = col.replace("air_quality_", "")
            pollutants[key] = pd.to_numeric(df[col], errors="coerce").to_numpy()

    # ---- Apparent temperature ----
    if "feels_like_celsius" in df.columns:
        app = pd.to_numeric(df["feels_like_celsius"], errors="coerce")
        app = app.where(app.notna(), pd.Series(heat_index_celsius(temp, rh), index=df.index))
        apparent = np.maximum(app.to_numpy(dtype=float), temp.to_numpy(dtype=float))
    else:
        apparent = heat_index_celsius(temp, rh)

    wc = wind_chill_celsius(temp, wind)

    # ---- Component scores ----
    f_heat = _points_heat(apparent)
    f_cold = _points_cold(wc)
    f_air = _points_air(epa.to_numpy(dtype=float), pollutants)
    f_uv = _points_uv(uv.to_numpy(dtype=float))

    # ---- SYNERGY (Heat × Air Quality) ----
    # Activation at 28.6°C (from Decision Tree root split)
    # NOTE: The synergy cap is 23 ED points (8 + 15), not 21 (comment corrected)
    both = (apparent > 28.6) & (f_air > 18)
    f_syn = np.where(
        both,
        8.0 + 15.0 * np.clip((f_air - 18) / 32.0, 0, 1),  # Max = 23
        0.0
    )
    
    # ---- Initial ED score ----
    score = np.clip(f_heat + f_cold + f_air + f_uv + f_syn, 0, 100)

    # ========================================================================
    # SAFETY FLOOR - Apply if any component is extreme
    # ========================================================================
    # Vectorized application: process each row
    adjusted_scores = []
    for i in range(len(score)):
        components = {
            "heat": float(f_heat[i]),
            "cold": float(f_cold[i]),
            "air": float(f_air[i]),
            "uv": float(f_uv[i]),
            "synergy": float(f_syn[i]),
        }
        adjusted = _apply_safety_floor(float(score[i]), components)
        adjusted_scores.append(adjusted)
    score = np.array(adjusted_scores)

    # ---- Output ----
    out = pd.DataFrame(index=df.index)
    out["apparent_temp_c"] = apparent
    out["wind_chill_c"] = wc
    out["ed_f_heat"] = f_heat
    out["ed_f_cold"] = f_cold
    out["ed_f_air"] = f_air
    out["ed_f_uv"] = f_uv
    out["ed_f_synergy"] = f_syn
    out["ed_score"] = score
    out["ed_category"] = _score_to_category(score)  # Single source of truth
    return out


def compute_ed_baseline(row: dict) -> dict:
    df = pd.DataFrame([row])
    r = compute_ed_baseline_frame(df).iloc[0]
    return {
        "ed_score": float(r["ed_score"]),
        "ed_category": str(r["ed_category"]),
        "apparent_temp_c": float(r["apparent_temp_c"]),
        "wind_chill_c": float(r["wind_chill_c"]),
        "components": {
            "heat": float(r["ed_f_heat"]),
            "cold": float(r["ed_f_cold"]),
            "air": float(r["ed_f_air"]),
            "uv": float(r["ed_f_uv"]),
            "synergy": float(r["ed_f_synergy"]),
        },
    }


if __name__ == "__main__":
    # Quick test (single-row, should work now)
    test = compute_ed_baseline({
        "temperature_celsius": 30,
        "humidity": 50,
        "wind_kph": 5,
        "uv_index": 0,
        "air_quality_us-epa-index": 4,
        "air_quality_PM2.5": 120,
    })
    print("Single-row test:", test)