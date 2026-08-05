"""
api/weather_api.py
==================
Standalone FastAPI server using Open-Meteo (free, no API key).
Combines weather + air quality data for the ED system.

Fixes applied:
    • Shared httpx.AsyncClient (connection pool reused)
    • Parallel weather + air-quality requests
    • Exponential-backoff retries on timeouts / 5xx
    • Split connect/read timeouts
    • Runs on port 8001 to avoid conflict with ed_api.py (port 8000)
"""
import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime
from typing import Dict, Optional, Tuple
import uvicorn

# ─── Config ──────────────────────────────────────────────────────────
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.75

# ─── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Exercise Danger Weather API (Open-Meteo)",
    description="Free weather and air quality data using Open-Meteo",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Shared HTTP client ──────────────────────────────────────────────
_http_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def _startup() -> None:
    global _http_client
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
    _http_client = httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        limits=limits,
        headers={"User-Agent": "ED-Weather-API/2.1"},
    )
    print("🚀 Shared httpx.AsyncClient ready.")


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


# ─── Simple Cache ────────────────────────────────────────────────────
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
    _cache[_get_cache_key(city)] = (data, datetime.now())


# ─── HTTP helper with retries ────────────────────────────────────────
async def fetch_with_retry(url: str, params: dict) -> dict:
    if _http_client is None:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")

    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await _http_client.get(url, params=params)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise httpx.HTTPStatusError(
                    f"Upstream {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                sleep_for = RETRY_BACKOFF_BASE * (2 ** attempt)
                print(f"⏱️  Retry {attempt + 1}/{MAX_RETRIES} for {url}: {exc}")
                await asyncio.sleep(sleep_for)
                continue
            break
        except httpx.HTTPError as exc:
            last_exc = exc
            break

    raise HTTPException(
        status_code=504,
        detail=f"External API failed after {MAX_RETRIES} attempts: {last_exc}",
    )


# ─── Geocoding ───────────────────────────────────────────────────────
async def get_coordinates(city: str) -> Tuple[float, float, str, str]:
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    data = await fetch_with_retry(GEOCODING_URL, params)
    if not data.get("results"):
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    r = data["results"][0]
    return float(r["latitude"]), float(r["longitude"]), r["name"], r.get("country", "")


# ─── Endpoints ───────────────────────────────────────────────────────
@app.get("/")
async def home():
    return {
        "message": "Exercise Danger Weather API (Open-Meteo)",
        "endpoints": {
            "/weather?city=London": "Get weather + air quality",
            "/health": "Health check",
        },
        "status": "running",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "http_client_ready": _http_client is not None,
    }


@app.get("/weather")
async def get_weather(city: str = Query(..., min_length=1)):
    """Get current weather + air quality for a city."""
    cached = _get_cached(city)
    if cached:
        return cached

    lat, lon, city_name, country = await get_coordinates(city)

    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "weather_code", "cloud_cover",
            "wind_speed_10m", "uv_index",
        ],
        "timezone": "auto",
    }
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone",
        ],
        "timezone": "auto",
    }

    # Parallel fetch
    w_task = asyncio.create_task(fetch_with_retry(WEATHER_URL, weather_params))
    aq_task = asyncio.create_task(fetch_with_retry(AIR_QUALITY_URL, aq_params))

    w_data = await w_task
    try:
        aq_data = await aq_task
        current_aq = aq_data.get("current", {})
    except HTTPException as e:
        print(f"⚠️ Air quality API unavailable: {e.detail}")
        current_aq = {}

    current_w = w_data.get("current", {})

    weather_codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }
    condition = weather_codes.get(current_w.get("weather_code", 0), "Unknown")

    result = {
        "city": city_name,
        "country": country,
        "temperature": current_w.get("temperature_2m"),
        "temperature_celsius": current_w.get("temperature_2m"),
        "humidity": current_w.get("relative_humidity_2m"),
        "wind_kph": current_w.get("wind_speed_10m"),
        "uv": current_w.get("uv_index"),
        "uv_index": current_w.get("uv_index"),
        "condition": condition,
        "air_quality": {
            "pm2_5": current_aq.get("pm2_5"),
            "pm10": current_aq.get("pm10"),
            "co": current_aq.get("carbon_monoxide"),
            "no2": current_aq.get("nitrogen_dioxide"),
            "so2": current_aq.get("sulphur_dioxide"),
            "o3": current_aq.get("ozone"),
        },
    }

    _set_cache(city, result)
    return result


# ─── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # NOTE: runs on 8001 to avoid conflict with ed_api.py (8000)
    uvicorn.run(
        "weather_api:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info",
    )
