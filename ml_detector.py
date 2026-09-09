import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class TelemetryAIDetector:
    """Uses Isolation Forest to detect multi-variable sensor anomalies in telemetry data."""

    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted = False

    def train(self, df):
        """Trains the ML model on nominal baseline telemetry."""
        features = ['altitude_km', 'velocity_kms', 'battery_voltage', 'temperature_c', 'signal_strength_dbm']
        self.model.fit(df[features])
        self.is_fitted = True

    def predict_anomaly(self, frame_df):
        """Predicts anomalies in current frame (-1 = Anomaly, 1 = Normal)."""
        if not self.is_fitted:
            return 1, 0.0

        features = ['altitude_km', 'velocity_kms', 'battery_voltage', 'temperature_c', 'signal_strength_dbm']
        prediction = self.model.predict(frame_df[features])
        score = self.model.decision_function(frame_df[features])

        return prediction[0], round(score[0], 3)