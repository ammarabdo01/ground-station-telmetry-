class TelemetryHealthAnalyzer:
    """Evaluates telemetry metrics against critical safety thresholds."""

    TEMP_WARN_LIMIT = 52.0
    TEMP_CRIT_LIMIT = 65.0
    VOLTAGE_LOW_LIMIT = 24.5
    RSSI_MIN_LIMIT = -85.0

    @classmethod
    def analyze_frame(cls, row):
        """Analyzes a single telemetry frame and returns health status."""
        status = "NOMINAL"
        warnings = []

        if row['temperature_c'] >= cls.TEMP_CRIT_LIMIT:
            status = "CRITICAL"
            warnings.append(f"OVERHEAT CRITICAL: {row['temperature_c']}°C")
        elif row['temperature_c'] >= cls.TEMP_WARN_LIMIT:
            if status != "CRITICAL":
                status = "WARNING"
            warnings.append(f"High Temp Warning: {row['temperature_c']}°C")

        if row['battery_voltage'] <= cls.VOLTAGE_LOW_LIMIT:
            status = "CRITICAL"
            warnings.append(f"BATTERY CRITICAL: {row['battery_voltage']}V")

        if row['signal_strength_dbm'] <= cls.RSSI_MIN_LIMIT:
            if status != "CRITICAL":
                status = "WARNING"
            warnings.append(f"Low Signal: {row['signal_strength_dbm']} dBm")

        return status, warnings