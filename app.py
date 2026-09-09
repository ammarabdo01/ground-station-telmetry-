import streamlit as st
import pandas as pd
import plotly.express as px
import time
from src.generator import generate_telemetry_data
from src.analyzer import TelemetryHealthAnalyzer
from src.ml_detector import TelemetryAIDetector

st.set_page_config(
    page_title="Ground Station Telemetry Dashboard",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Ground Station Telemetry Dashboard")
st.markdown("*Real-time Spacecraft Health Signal Monitoring, AI Anomaly Detection & Telemetry Replay Engine*")

# Sidebar Controls
st.sidebar.header("🕹️ Mission Control")
playback_speed = st.sidebar.slider("Replay Delay (seconds)", 0.05, 1.0, 0.2)
generate_new = st.sidebar.button("🔄 Generate New Flight Simulation")

# Initialize Session State
if 'data' not in st.session_state or generate_new:
    st.session_state.data = generate_telemetry_data(num_samples=120)
    st.session_state.step = 0

    st.session_state.ai_detector = TelemetryAIDetector(contamination=0.1)
    st.session_state.ai_detector.train(st.session_state.data.iloc[:40])

df = st.session_state.data

# Sidebar Report Export
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Mission Reports")
anomalies_df = df[df['temperature_c'] >= 52.0]
if not anomalies_df.empty:
    st.sidebar.download_button(
        label="Download Anomaly Log (CSV)",
        data=anomalies_df.to_csv(index=False),
        file_name="telemetry_anomaly_report.csv",
        mime="text/csv"
    )

kpi_container = st.empty()
chart_container = st.container()

run_sim = st.sidebar.checkbox("▶️ Start Telemetry Live Stream", value=True)

if run_sim:
    current_step = st.session_state.step
    if current_step < len(df):
        sub_df = df.iloc[:current_step + 1]
        latest_frame = sub_df.iloc[[-1]]
        latest_row = sub_df.iloc[-1]

        status, warnings = TelemetryHealthAnalyzer.analyze_frame(latest_row)
        ai_pred, ai_score = st.session_state.ai_detector.predict_anomaly(latest_frame)

        # Dashboard KPIs
        with kpi_container.container():
            c1, c2, c3, c4, c5, c6 = st.columns(6)

            c1.metric("Rule Health", status)
            c2.metric("AI Detector", "⚠️ ANOMALY" if ai_pred == -1 else "✅ NORMAL")
            c3.metric("Altitude", f"{latest_row['altitude_km']} km")
            c4.metric("Velocity", f"{latest_row['velocity_kms']} km/s")
            c5.metric("Temp", f"{latest_row['temperature_c']} °C")
            c6.metric("Battery", f"{latest_row['battery_voltage']} V")

            if warnings:
                for w in warnings:
                    st.warning(f"⚠️ Rule Warning: {w}")
            if ai_pred == -1:
                st.error(f"🚨 AI Alert: Multi-variable Anomaly Detected! (Score: {ai_score})")

        # Interactive Charts
        with chart_container:
            col1, col2 = st.columns(2)

            with col1:
                fig_temp = px.line(sub_df, x='timestamp', y='temperature_c',
                                   title="Thermal Telemetry (°C)", markers=True)
                fig_temp.add_hline(y=52.0, line_dash="dash", line_color="orange")
                fig_temp.add_hline(y=65.0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_temp, use_container_width=True)

            with col2:
                fig_batt = px.line(sub_df, x='timestamp', y='battery_voltage',
                                   title="Power Telemetry (Voltage)", markers=True)
                fig_batt.add_hline(y=24.5, line_dash="dash", line_color="red")
                st.plotly_chart(fig_batt, use_container_width=True)

            fig_3d = px.line_3d(
                sub_df,
                x='altitude_km',
                y='velocity_kms',
                z='temperature_c',
                color='temperature_c',
                title="3D Telemetry State-Space Trajectory",
                markers=True
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        st.session_state.step += 1
        time.sleep(playback_speed)
        st.rerun()
    else:
        st.success("🏁 Telemetry Playback Completed.")