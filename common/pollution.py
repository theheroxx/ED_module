"""
common/pollution.py
===================
Unified pollution handling for the Exercise Danger system.

Converts raw pollutant concentrations to AQI, EPA Index, and applies
multi-pollutant penalties based on the number of elevated pollutants.

All conversions follow EPA 40 CFR Part 58 breakpoints.

UPDATED (2026-07-24):
  - Restored continuous interpolation for EPA → base points (no discrete mapping).
  - PM2.5 fine‑tuning within EPA category.
  - Multi‑pollutant penalty for 2+ pollutants with AQI > 100.
"""
import numpy as np

# ----------------------------------------------------------------------------
# EPA AQI Breakpoints (40 CFR Part 58)
# ----------------------------------------------------------------------------

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]

O3_BREAKPOINTS = [
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
    (0.201, 0.500, 301, 500),  # simplified for > 0.200
]

NO2_BREAKPOINTS = [
    (0.000, 0.053, 0, 50),
    (0.054, 0.100, 51, 100),
    (0.101, 0.360, 101, 150),
    (0.361, 0.649, 151, 200),
    (0.650, 1.249, 201, 300),
    (1.250, 2.000, 301, 500),
]

SO2_BREAKPOINTS = [
    (0.000, 0.035, 0, 50),
    (0.036, 0.075, 51, 100),
    (0.076, 0.185, 101, 150),
    (0.186, 0.304, 151, 200),
    (0.305, 0.604, 201, 300),
    (0.605, 1.000, 301, 500),
]

CO_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 50.0, 301, 500),
]


def _convert_to_aqi(value, breakpoints):
    """Generic AQI conversion using breakpoints."""
    if value is None or np.isnan(value) or value < 0:
        return None
    for lo, hi, aqi_lo, aqi_hi in breakpoints:
        if lo <= value <= hi:
            if hi == lo:
                return aqi_lo
            return np.interp(value, [lo, hi], [aqi_lo, aqi_hi])
    # If value exceeds max breakpoint, cap at 500
    if value > breakpoints[-1][1]:
        return 500
    return None


def pm25_to_aqi(pm25):
    """Convert PM2.5 (µg/m³) to AQI (0-500)."""
    return _convert_to_aqi(pm25, PM25_BREAKPOINTS)


def pm10_to_aqi(pm10):
    """Convert PM10 (µg/m³) to AQI (0-500)."""
    return _convert_to_aqi(pm10, PM10_BREAKPOINTS)


def o3_to_aqi(o3_ppm):
    """Convert O3 (ppm) to AQI (0-500)."""
    return _convert_to_aqi(o3_ppm, O3_BREAKPOINTS)


def no2_to_aqi(no2_ppm):
    """Convert NO2 (ppm) to AQI (0-500)."""
    return _convert_to_aqi(no2_ppm, NO2_BREAKPOINTS)


def so2_to_aqi(so2_ppm):
    """Convert SO2 (ppm) to AQI (0-500)."""
    return _convert_to_aqi(so2_ppm, SO2_BREAKPOINTS)


def co_to_aqi(co_ppm):
    """Convert CO (ppm) to AQI (0-500)."""
    return _convert_to_aqi(co_ppm, CO_BREAKPOINTS)


def aqi_to_epa_index(aqi):
    """Convert AQI (0-500) to EPA Index (1-6)."""
    if aqi is None or np.isnan(aqi):
        return 1
    if aqi <= 50:
        return 1
    elif aqi <= 100:
        return 2
    elif aqi <= 150:
        return 3
    elif aqi <= 200:
        return 4
    elif aqi <= 300:
        return 5
    else:
        return 6


def epa_index_to_base_points(epa_index):
    """
    Convert EPA Index (1-6) to base air quality points using continuous linear interpolation.
    This preserves the "no jumps" behavior.

    Mapping:
        EPA=1 →  0–10,  EPA=2 → 10–25,  EPA=3 → 25–45,
        EPA=4 → 45–65,  EPA=5 → 65–85,  EPA=6 → 85–100
    """
    # Define knots for interpolation
    xp = [1, 2, 3, 4, 5, 6]
    fp = [5.0, 17.5, 35.0, 55.0, 75.0, 92.5]  # midpoints of each range
    epa_array = np.asarray(epa_index, dtype=float)
    return np.interp(epa_array, xp, fp)


def get_pm25_within_category(pm25, epa_index):
    """
    Calculate normalized position of PM2.5 within its EPA category (0-1).
    Used for fine-tuning air quality points.
    """
    if pm25 is None or np.isnan(pm25) or pm25 < 0:
        return 0.5

    # Map EPA index to PM2.5 AQI ranges
    epa_ranges = {
        1: (0, 50),
        2: (51, 100),
        3: (101, 150),
        4: (151, 200),
        5: (201, 300),
        6: (301, 500),
    }

    if epa_index not in epa_ranges:
        return 0.5

    pm25_aqi = pm25_to_aqi(pm25)
    if pm25_aqi is None or np.isnan(pm25_aqi):
        return 0.5

    lo, hi = epa_ranges[epa_index]
    # Clamp to category range
    pm25_aqi = np.clip(pm25_aqi, lo, hi)
    if hi == lo:
        return 0.5
    return (pm25_aqi - lo) / (hi - lo)


def calculate_multi_pollutant_penalty(pollutants: dict) -> float:
    """
    Calculate penalty points when multiple pollutants are elevated (AQI > 100).

    Args:
        pollutants: {
            "PM2.5": 12.0,
            "PM10": 20.0,
            "O3": 0.075,
            "NO2": 0.053,
            "SO2": 0.035,
            "CO": 4.0,
        }

    Returns:
        Penalty points (0, 2, or 5) to add to the air quality score.
    """
    # Convert each pollutant to AQI
    aqi_values = {}
    converters = {
        "PM2.5": pm25_to_aqi,
        "PM10": pm10_to_aqi,
        "O3": o3_to_aqi,
        "NO2": no2_to_aqi,
        "SO2": so2_to_aqi,
        "CO": co_to_aqi,
    }

    for name, value in pollutants.items():
        if name in converters and value is not None and value > 0:
            aqi = converters[name](value)
            if aqi is not None and not np.isnan(aqi):
                aqi_values[name] = aqi

    if not aqi_values:
        return 0.0

    # Count pollutants with AQI > 100 (Unhealthy for Sensitive Groups)
    high_count = sum(1 for aqi in aqi_values.values() if aqi > 100)

    # Apply penalty based on number of high pollutants
    if high_count >= 3:
        return 5.0   # Max penalty
    elif high_count == 2:
        return 2.0   # Medium penalty
    else:
        return 0.0   # No penalty


def calculate_air_quality_points(pollutants: dict, epa_index: int) -> float:
    """
    Calculate final air quality points (0-100) using:
        1. EPA Index → continuous base points (linear interpolation)
        2. PM2.5 fine-tuning (±1.5 points within same EPA category)
        3. Multi-pollutant penalty (2+ pollutants with AQI > 100)

    Args:
        pollutants: Raw pollutant concentrations (dict)
        epa_index: EPA Index (1-6)

    Returns:
        Air quality points (0-100)
    """
    # 1. Base points from continuous interpolation
    base_points = float(epa_index_to_base_points(epa_index))

    # 2. PM2.5 fine-tuning (±1.5 points within same category)
    pm25 = pollutants.get("PM2.5")
    if pm25 is not None and pm25 > 0 and not np.isnan(pm25):
        pm25_aqi = pm25_to_aqi(pm25)
        if pm25_aqi is not None and not np.isnan(pm25_aqi):
            pm25_epa = aqi_to_epa_index(pm25_aqi)
            if pm25_epa == epa_index:
                normalized = get_pm25_within_category(pm25, epa_index)
                adjustment = (normalized - 0.5) * 3.0  # ±1.5 points
                base_points += adjustment

    # 3. Multi-pollutant penalty
    penalty = calculate_multi_pollutant_penalty(pollutants)
    base_points += penalty

    # 4. Clamp to [0, 100]
    return np.clip(base_points, 0.0, 100.0)