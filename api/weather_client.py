"""
api/weather_client.py
=====================
Client for the ED weather API.

Fixes applied:
    • Configurable base URL via env var (ED_API_URL)
    • Session reuse (connection pooling)
    • Retry with exponential backoff on timeouts / 5xx
    • Longer, split connect/read timeouts
    • Clear, actionable error messages
"""
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, Optional

# Point this at wherever ed_api.py is running.
# Override with:  export ED_API_URL=http://my-host:8000
API_BASE_URL = os.environ.get("ED_API_URL", "http://127.0.0.1:8000")

# (connect timeout, read timeout) — connect fails fast, read is patient
REQUEST_TIMEOUT = (5, 45)
MAX_RETRIES = 3


def _make_session() -> requests.Session:
    """Session with automatic retries + connection pooling."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.75,             # 0.75, 1.5, 3.0 s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# One shared session for the lifetime of the client
_session = _make_session()


def get_weather(city: str, timeout: Optional[tuple] = None) -> Dict[str, Any]:
    """
    Get weather data from the ED API.

    Raises:
        requests.exceptions.ConnectionError – server not reachable
        requests.exceptions.Timeout         – server didn't answer in time
        requests.exceptions.HTTPError       – 4xx/5xx after retries
    """
    url = f"{API_BASE_URL.rstrip('/')}/weather"
    t = timeout or REQUEST_TIMEOUT

    try:
        response = _session.get(url, params={"city": city}, timeout=t)
    except requests.exceptions.ConnectionError as e:
        raise requests.exceptions.ConnectionError(
            f"Cannot reach ED API at {API_BASE_URL}. "
            f"Is the server running?  Start it with:\n"
            f"    python -m uvicorn api.ed_api:app --host 0.0.0.0 --port 8000\n"
            f"Original error: {e}"
        ) from e
    except requests.exceptions.Timeout as e:
        raise requests.exceptions.Timeout(
            f"ED API timed out after {t[1]}s while fetching weather for '{city}'. "
            f"Open-Meteo may be slow — try again in a moment."
        ) from e

    if not response.ok:
        raise requests.exceptions.HTTPError(
            f"ED API returned {response.status_code} for city='{city}': "
            f"{response.text[:300]}"
        )
    return response.json()


def get_prediction_for_city(
    city: str,
    cluster_id: Optional[int] = None,
    anomaly_flag: bool = False,
    timeout: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Convenience: fetch weather + ED prediction in a single call."""
    url = f"{API_BASE_URL.rstrip('/')}/predict_city"
    params: Dict[str, Any] = {"city": city, "anomaly_flag": str(anomaly_flag).lower()}
    if cluster_id is not None:
        params["cluster_id"] = cluster_id

    response = _session.get(url, params=params, timeout=timeout or REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def health_check() -> bool:
    """Return True if the ED API is reachable and healthy."""
    try:
        r = _session.get(f"{API_BASE_URL.rstrip('/')}/health", timeout=(3, 5))
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False
