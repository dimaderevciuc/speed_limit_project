import pandas as pd
from pathlib import Path


def load_data():
    # Resolve data directory relative to project root
    base = Path(__file__).resolve().parents[1]
    data_dir = base / 'data'

    sensors = pd.read_csv(data_dir / 'sensors.csv', sep=';')
    # Filename in repository is 'sensor_type.csv' (singular)
    sensor_types = pd.read_csv(data_dir / 'sensor_type.csv', sep=';')
    accidents = pd.read_csv(data_dir / 'accidents.csv', sep=';')
    readings = pd.read_csv(data_dir / 'readings.csv', sep=';')

    return sensors, sensor_types, accidents, readings