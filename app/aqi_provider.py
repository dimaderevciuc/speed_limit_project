import json
import urllib.request
import urllib.parse
import logging
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


def pm25_to_aqi(pm25: float) -> int:
    """Convert PM2.5 concentration (µg/m3) to US EPA AQI using breakpoints."""
    if pm25 is None:
        return 0
    # Breakpoints
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for (clow, chigh, ilow, ihigh) in breakpoints:
        if pm25 >= clow and pm25 <= chigh:
            aqi = (ihigh - ilow) / (chigh - clow) * (pm25 - clow) + ilow
            return int(round(aqi))
    # Above 500.4
    return 500


def _fetch_open_meteo_pm25(lat: float, lon: float, timeout: int = 5):
    """Fetch PM2.5 from Open-Meteo air quality API. Returns float or None."""
    base = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "hourly": "pm2_5",
        "timezone": "UTC",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
            data = json.loads(raw)
            # Expect structure: data['hourly']['pm2_5'] -> list
            hourly = data.get("hourly", {})
            pm25_list = hourly.get("pm2_5")
            if pm25_list and len(pm25_list) > 0:
                # Use the latest available value
                val = pm25_list[0]
                if val is None:
                    return None
                return float(val)
    except Exception as e:
        logger.warning("Open-Meteo fetch failed: %s", e)
    return None


def _estimate_pm25_from_csv(data_dir: Path):
    """Attempt a naive estimation of PM2.5 from local `readings.csv`. This is a best-effort fallback.
    Returns median value from numeric readings in a plausible PM2.5 range (0..500) or None.
    """
    try:
        f = data_dir / "readings.csv"
        if not f.exists():
            return None
        vals = []
        with f.open("r", encoding="utf-8") as fh:
            # Skip header
            next(fh)
            for line in fh:
                parts = line.strip().split(";")
                if len(parts) < 7:
                    continue
                raw_val = parts[6].replace(",", ".")
                try:
                    v = float(raw_val)
                except Exception:
                    continue
                # Keep plausible PM2.5 ranges
                if 0.0 < v <= 500.0:
                    vals.append(v)
        if not vals:
            return None
        # Use median to reduce outliers
        return float(statistics.median(vals))
    except Exception as e:
        logger.warning("CSV PM2.5 estimation failed: %s", e)
        return None


def get_current_aqi(lat: float = None, lon: float = None):
    """Get current AQI for given lat/lon using Open-Meteo.

    Returns a tuple: (aqi:int, source:str) where source is one of
    'api' (Open-Meteo), 'csv' (local CSV estimate), or 'default'.
    """
    # If lat/lon not provided, use a sensible default (Berlin)
    if lat is None or lon is None:
        lat = float("52.52")
        lon = float("13.405")

    pm25 = _fetch_open_meteo_pm25(lat, lon)
    if pm25 is not None:
        aqi = pm25_to_aqi(pm25)
        logger.info("Fetched pm2.5=%s -> AQI=%s from Open-Meteo", pm25, aqi)
        return aqi, "api"

    # Fallback: try to estimate from local CSV
    base = Path(__file__).resolve().parents[1]
    est = _estimate_pm25_from_csv(base / "data")
    if est is not None:
        aqi = pm25_to_aqi(est)
        logger.info("Estimated pm2.5=%s -> AQI=%s from local CSV", est, aqi)
        return aqi, "csv"

    # Final fallback: default AQI
    logger.warning("Could not obtain AQI from API or CSV — using default AQI=50")
    return 50, "default"
