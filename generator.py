import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_telemetry_data(num_samples=120):
    """Generates simulated spacecraft telemetry time-series dataset."""
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_samples)]

    altitude = 400 + np.cumsum(np.random.normal(0, 0.5, num_samples))
    velocity = 7.6 + np.random.normal(0, 0.05, num_samples)
    battery_v = 28.0 + np.random.normal(0, 0.2, num_samples)
    temp_c = 45.0 + np.random.normal(0, 1.2, num_samples)
    rssi = -70 + np.random.normal(0, 2.0, num_samples)

    # Inject anomalies for test replay
    temp_c[50:65] += np.linspace(5, 25, 15)
    battery_v[90:105] -= np.linspace(2, 6, 15)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'altitude_km': np.round(altitude, 2),
        'velocity_kms': np.round(velocity, 3),
        'battery_voltage': np.round(battery_v, 2),
        'temperature_c': np.round(temp_c, 2),
        'signal_strength_dbm': np.round(rssi, 1)
    })
    return df


if __name__ == "__main__":
    df = generate_telemetry_data()
    df.to_csv("data/telemetry_dataset.csv", index=False)
    print("✅ Dataset generated successfully at data/telemetry_dataset.csv")