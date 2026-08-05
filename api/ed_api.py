"""
api/ed_api.py
=============
FastAPI server for Exercise Danger Prediction (Grid-Based GNN).

Endpoints:
    /predict          → POST/GET with weather params → ED score
    /predict_city     → GET with city name → auto-fetch weather → ED score
    /weather          → GET raw weather + AQI for a city
    /health           → Health check
    /bias_map         → Get current GNN bias map

Fixes applied:
    • Blocking `requests` → async `httpx` (no event-loop blocking)
    • Single shared AsyncClient with connection pool
    • Parallel weather + air-quality fetch via asyncio.gather
    • Exponential-backoff retries on timeouts / 5xx
    • Longer, split timeouts (connect vs read)
    • Cache is checked BEFORE any network I/O
    • Gas pollutants (CO, NO2, SO2, O3) are converted from µg/m³ → ppm
      at the server, so the Streamlit app receives clean ppm values.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import numpy as np

# Import the math model
from math_model.math_model import ExerciseDangerMathModel
from common.pollution import (
    pm25_to_aqi, pm10_to_aqi, o3_to_aqi, no2_to_aqi,
    so2_to_aqi, co_to_aqi, aqi_to_epa_index
)

# ─── Open-Meteo endpoints ────────────────────────────────────────────
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Split timeouts: fail fast on connect, be patient on read
HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.75  # seconds; grows exponentially

# ─── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Exercise Danger Prediction API (Grid-Based GNN)",
    description="ED scores with grid-based GNN regional adjustments",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model instance (loads grid-based GNN bias map) ──────────────────
_model = ExerciseDangerMathModel()
print(f"✅ ED model initialized. Bias map: {_model.bias_map}")

# ─── Shared HTTP client (connection pool reused across requests) ─────
_http_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def _startup() -> None:
    global _http_client
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
    _http_client = httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        limits=limits,
        headers={"User-Agent": "ED-API/2.1"},
        http2=False,
    )
    print("🚀 Shared httpx.AsyncClient ready.")


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        print("👋 httpx.AsyncClient closed.")


# ─── Simple cache for weather ────────────────────────────────────────
_cache: Dict[str, tuple] = {}
CACHE_TTL = 600  # 10 minutes


def _get_cache_key(city: str) -> str:
    return city.lower().strip()


def _get_cached(city: str) -> Optional[Dict]:
    key = _get_cache_key(city)
    if key in _cache:
        data, timestamp = _cache[key]
        if (datetime.now() - timestamp).total_seconds() < CACHE_TTL:
            return data
    return None


def _set_cache(city: str, data: Dict) -> None:
    key = _get_cache_key(city)
    _cache[key] = (data, datetime.now())


# ─── HTTP helper with retries ────────────────────────────────────────
async def _fetch_json(url: str, params: dict) -> dict:
    """
    GET `url` with `params` using the shared async client.
    Retries on timeouts, connection errors and 5xx responses with
    exponential backoff.
    """
    if _http_client is None:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")

    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await _http_client.get(url, params=params)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise httpx.HTTPStatusError(
                    f"Upstream {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                sleep_for = RETRY_BACKOFF_BASE * (2 ** attempt)
                print(f"⏱️  Attempt {attempt + 1}/{MAX_RETRIES} for {url} failed: {exc}. "
                      f"Retrying in {sleep_for:.1f}s...")
                await asyncio.sleep(sleep_for)
                continue
            break
        except httpx.HTTPError as exc:
            last_exc = exc
            break

    detail = f"Upstream request to {url} failed after {MAX_RETRIES} attempts: {last_exc}"
    raise HTTPException(status_code=504, detail=detail)


# ─── Geocoding ───────────────────────────────────────────────────────
async def get_coordinates(city: str) -> Tuple[float, float, str, str]:
    """Convert city name → (lat, lon, display_name, country)."""
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    data = await _fetch_json(GEOCODING_URL, params)
    if not data.get("results"):
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    result = data["results"][0]
    return (
        float(result["latitude"]),
        float(result["longitude"]),
        result["name"],
        result.get("country", ""),
    )


# ─── Unit conversion helper: µg/m³ → ppm ──────────────────────────
def ugm3_to_ppm(value: Optional[float], molecular_weight: float) -> Optional[float]:
    """
    Convert concentration from µg/m³ to ppm at 25°C, 1 atm.
    Uses molar volume: 24.45 L/mol.
    ppm = (µg/m³ * 24.45) / (molecular_weight * 1000.0)
    """
    if value is None:
        return None
    return float(value) * 24.45 / (molecular_weight * 1000.0)


# ─── Weather fetcher ─────────────────────────────────────────────────
# ─── Weather fetcher ─────────────────────────────────────────────────
async def fetch_weather(city: str) -> Dict[str, Any]:
    """Get current weather + air quality for a city (cached, parallel)."""
    cached = _get_cached(city)
    if cached:
        return cached

    lat, lon, city_name, country = await get_coordinates(city)

    w_params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
            "uv_index",
        ],
        "timezone": "auto",
    }
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ],
        "timezone": "auto",
    }

    # Fire both requests in parallel
    w_task = asyncio.create_task(_fetch_json(WEATHER_URL, w_params))
    aq_task = asyncio.create_task(_fetch_json(AIR_QUALITY_URL, aq_params))

    w_data = await w_task
    try:
        aq_data = await aq_task
        aq_current = aq_data.get("current", {})
    except HTTPException as e:
        print(f"⚠️ Air quality API unavailable: {e.detail}")
        aq_current = {}

    # ─── DEBUG: Print raw AQ values ──────────────────────────────
    print(f"🔍 Raw AQ data: {aq_current}")

    current_w = w_data.get("current", {})

    # ─── Extract and convert ──────────────────────────────────────
    raw_o3 = aq_current.get("ozone")
    raw_co = aq_current.get("carbon_monoxide")
    raw_no2 = aq_current.get("nitrogen_dioxide")
    raw_so2 = aq_current.get("sulphur_dioxide")

    result = {
        "city": city_name,
        "country": country,
        "temperature": current_w.get("temperature_2m"),
        "temperature_celsius": current_w.get("temperature_2m"),
        "humidity": current_w.get("relative_humidity_2m"),
        "wind_kph": current_w.get("wind_speed_10m"),
        "uv": current_w.get("uv_index"),
        "uv_index": current_w.get("uv_index"),
        "condition": "Unknown",  # You can add weather code mapping here
        "air_quality": {
            "pm2_5": aq_current.get("pm2_5"),          # µg/m³
            "pm10": aq_current.get("pm10"),            # µg/m³
            "co": ugm3_to_ppm(raw_co, 28.01),          # → ppm
            "no2": ugm3_to_ppm(raw_no2, 46.01),        # → ppm
            "so2": ugm3_to_ppm(raw_so2, 64.07),        # → ppm
            "o3": ugm3_to_ppm(raw_o3, 48.00),          # → ppm
        },
    }

    _set_cache(city, result)
    return result


# ─── Endpoints ───────────────────────────────────────────────────────
@app.get("/")
async def home():
    return {
        "message": "Exercise Danger Prediction API (Grid-Based GNN)",
        "endpoints": {
            "/weather?city=London": "Get raw weather + AQI data",
            "/predict": "POST with weather params → ED score",
            "/predict_city?city=London": "Auto-fetch weather → ED score",
            "/bias_map": "Get current GNN bias map",
            "/health": "Health check",
        },
        "model_type": "Grid-based GNN (1938 nodes)",
        "bias_map": _model.bias_map,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "bias_map_loaded": bool(_model.bias_map),
        "clusters": len(_model.bias_map) if _model.bias_map else 0,
        "http_client_ready": _http_client is not None,
    }


@app.get("/bias_map")
async def get_bias_map():
    return {
        "bias_map": _model.bias_map,
        "model_type": "Grid-based GNN (1938 nodes)",
        "clusters": len(_model.bias_map) if _model.bias_map else 0,
    }


@app.get("/weather")
async def weather_endpoint(city: str = Query(..., min_length=1)):
    """Get current weather + air quality for a city."""
    return await fetch_weather(city)


@app.post("/predict")
async def predict_endpoint(
    temperature_celsius: float = Query(..., description="°C"),
    humidity: float = Query(..., description="%"),
    wind_kph: float = Query(..., description="km/h"),
    uv_index: float = Query(..., description="0-15"),
    air_quality_us_epa_index: int = Query(..., ge=1, le=6, description="EPA Index 1-6"),
    air_quality_PM2_5: Optional[float] = Query(None, description="µg/m³"),
    air_quality_PM10: Optional[float] = Query(None, description="µg/m³"),
    air_quality_Ozone: Optional[float] = Query(None, description="ppm"),
    air_quality_Nitrogen_dioxide: Optional[float] = Query(None, description="ppm"),
    air_quality_Sulphur_dioxide: Optional[float] = Query(None, description="ppm"),
    air_quality_Carbon_Monoxide: Optional[float] = Query(None, description="ppm"),
    cluster_id: Optional[int] = Query(None, ge=0, le=5),
    anomaly_flag: bool = Query(False, description="Force ED=100"),
):
    """Direct ED prediction from provided weather and pollutant values."""
    result = await asyncio.to_thread(
        _model.predict,
        temperature_celsius=temperature_celsius,
        humidity=humidity,
        wind_kph=wind_kph,
        uv_index=uv_index,
        air_quality_us_epa_index=air_quality_us_epa_index,
        air_quality_PM2_5=air_quality_PM2_5,
        air_quality_PM10=air_quality_PM10,
        air_quality_Ozone=air_quality_Ozone,
        air_quality_Nitrogen_dioxide=air_quality_Nitrogen_dioxide,
        air_quality_Sulphur_dioxide=air_quality_Sulphur_dioxide,
        air_quality_Carbon_Monoxide=air_quality_Carbon_Monoxide,
        cluster_id=cluster_id,
        anomaly_flag=anomaly_flag,
    )
    result["model_type"] = "Grid-based GNN (1938 nodes)"
    return result


@app.get("/predict_city")
async def predict_city_endpoint(
    city: str = Query(..., min_length=1),
    cluster_id: Optional[int] = Query(None, ge=0, le=5),
    anomaly_flag: bool = Query(False),
):
    """Fetch weather for a city and return ED prediction."""
    try:
        weather = await fetch_weather(city)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {e}")

    temp = weather.get("temperature_celsius")
    hum = weather.get("humidity")
    wind = weather.get("wind_kph")
    uv = weather.get("uv_index")
    aq = weather.get("air_quality", {}) or {}

    aqi_values: Dict[str, float] = {}
    if aq.get("pm2_5") is not None:
        aqi_values["PM2.5"] = pm25_to_aqi(aq["pm2_5"])
    if aq.get("pm10") is not None:
        aqi_values["PM10"] = pm10_to_aqi(aq["pm10"])
    if aq.get("o3") is not None:
        aqi_values["O3"] = o3_to_aqi(aq["o3"])
    if aq.get("no2") is not None:
        aqi_values["NO2"] = no2_to_aqi(aq["no2"])
    if aq.get("so2") is not None:
        aqi_values["SO2"] = so2_to_aqi(aq["so2"])
    if aq.get("co") is not None:
        aqi_values["CO"] = co_to_aqi(aq["co"])

    epa_index = aqi_to_epa_index(max(aqi_values.values())) if aqi_values else 1

    result = await asyncio.to_thread(
        _model.predict,
        temperature_celsius=temp,
        humidity=hum,
        wind_kph=wind,
        uv_index=uv,
        air_quality_us_epa_index=epa_index,
        air_quality_PM2_5=aq.get("pm2_5"),
        air_quality_PM10=aq.get("pm10"),
        air_quality_Ozone=aq.get("o3"),
        air_quality_Nitrogen_dioxide=aq.get("no2"),
        air_quality_Sulphur_dioxide=aq.get("so2"),
        air_quality_Carbon_Monoxide=aq.get("co"),
        cluster_id=cluster_id,
        anomaly_flag=anomaly_flag,
    )

    result["weather"] = {
        "city": weather["city"],
        "country": weather["country"],
        "condition": weather["condition"],
        "temperature_celsius": temp,
        "humidity": hum,
        "wind_kph": wind,
        "uv_index": uv,
    }
    result["model_type"] = "Grid-based GNN (1938 nodes)"
    return result


# ─── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.ed_api:app", host="0.0.0.0", port=8000, reload=False)