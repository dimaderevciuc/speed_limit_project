import sys
from pathlib import Path

# Ensure project root is on sys.path so `from app...` imports work when
# running `python app/main.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

# Keep logging quiet for terminal output (show only errors). Detailed logs
# are still available if the user raises the level for debugging.
logging.basicConfig(level=logging.ERROR)

from app.decision_module import calculate_speed_limit
from app.data_loader import load_data
from app.aqi_provider import get_current_aqi

if __name__ == "__main__":
    sensors, sensor_types, accidents, readings = load_data()

    # Fetch live AQI (Open-Meteo) with CSV fallback. Coordinates can be set
    # via env vars `AQI_LAT` and `AQI_LON`. Defaults to Berlin.
    import os
    try:
        lat = float(os.getenv("AQI_LAT", "52.52"))
        lon = float(os.getenv("AQI_LON", "13.405"))
    except Exception:
        lat, lon = 52.52, 13.405

    aqi_now, aqi_source = get_current_aqi(lat, lon)

    # Build a small set of cases around current AQI to show variation
    deltas = [-20, 0, 20, 80]
    cases = []
    for d in deltas:
        v = max(0, int(aqi_now + d))
        cases.append({"weather": "good", "darkness": False, "air_quality": v})

    # Add one NN-driven adverse weather case using the current AQI as context
    cases.insert(0, {"weather": "bad", "darkness": True, "air_quality": aqi_now})

    # Print only the input and the final output (minimal)
    for i, c in enumerate(cases, 1):
        speed, _decision_source = calculate_speed_limit(c["weather"], c["darkness"], c["air_quality"])
        final_speed = int(round(speed))
        # Only print the input fields and the resulting speed
        print(f"AQI={c['air_quality']} Weather={c['weather']} Darkness={c['darkness']} -> {final_speed} km/h")
