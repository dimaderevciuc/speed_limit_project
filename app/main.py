import sys
from pathlib import Path

# Ensure project root is on sys.path so `from app...` imports work when
# running `python app/main.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.decision_module import calculate_speed_limit
from app.data_loader import load_data

if __name__ == "__main__":
    sensors, sensor_types, accidents, readings = load_data()

    # New test cases focused on verifying strict integer LLM responses.
    # We include a NN-driven case (bad weather/dark) and several AQI cases
    # where the LLM should respond with an exact integer reduction (single km).
    cases = [
        {"weather": "bad", "darkness": True, "air_quality": 40},   # NN path
        {"weather": "good", "darkness": False, "air_quality": 55},  # LLM path
        {"weather": "good", "darkness": False, "air_quality": 99},  # LLM path
        {"weather": "good", "darkness": False, "air_quality": 120}, # LLM path
        {"weather": "good", "darkness": False, "air_quality": 175}, # LLM path
        {"weather": "good", "darkness": False, "air_quality": 30},  # base
    ]

    for i, c in enumerate(cases, 1):
        speed, source = calculate_speed_limit(c["weather"], c["darkness"], c["air_quality"])
        src_text = {
            'nn': 'NeuralNet',
            'llm': 'LLM (exact integer)',
            'fallback': 'Deterministic fallback',
            'base': 'Base speed (no change)'
        }.get(source, source)

        print(f"Case {i}: {c} ➜ Speed limit: {speed:.2f} km/h  [{src_text}]")
