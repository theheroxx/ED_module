# APP.PY for testing the ED module
import streamlit as st
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common.pollution import (
    pm25_to_aqi,
    pm10_to_aqi,
    o3_to_aqi,
    no2_to_aqi,
    so2_to_aqi,
    co_to_aqi,
    aqi_to_epa_index,
    calculate_multi_pollutant_penalty,
)

from math_model.math_model import ExerciseDangerMathModel
from api.weather_client import get_weather, health_check, API_BASE_URL

model = ExerciseDangerMathModel()

def apparent_temp(temp, humidity):
    if temp < 20:
        return temp
    hi = temp + 0.33 * humidity * 0.01 - 0.7 * (temp - 20)
    return max(temp, hi)

def category_color(category):
    mapping = {
        "ED_VERY_SAFE": "green",
        "ED_MODERATE_SAFE": "lightgreen",
        "ED_CAUTION": "orange",
        "ED_DANGEROUS": "red",
        "ED_VERY_DANGEROUS": "darkred",
    }
    return mapping.get(category, "gray")

def category_clean(category):
    return category.replace("ED_", "").replace("_", " ").title()

EPA_LABELS = {
    1: "Good",
    2: "Moderate",
    3: "Unhealthy for Sensitive",
    4: "Unhealthy",
    5: "Very Unhealthy",
    6: "Hazardous",
}

st.set_page_config(
    page_title="Exercise Danger Predictor",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title("Exercise Danger Prediction System")
    st.markdown("### Medically Grounded Baseline + Grid-Based GNN Regional Adjustments")
    st.markdown("---")

    with st.sidebar:
        st.markdown("### Backend Status")
        st.caption(f"API URL: `{API_BASE_URL}`")
        if health_check():
            st.success("API is reachable")
        else:
            st.error(
                "API is NOT reachable\n\n"
                "Start it with:\n\n"
                "`python -m uvicorn api.ed_api:app --host 0.0.0.0 --port 8000`"
            )

    if model.bias_map:
        st.info(f"GNN regional adjustments loaded for {len(model.bias_map)} clusters (grid-based)")
    else:
        st.warning("No GNN regional adjustments loaded. Using baseline only.")

    st.subheader("Environmental Conditions")

    col_city, col_button = st.columns([3, 1])
    with col_city:
        city = st.text_input("City name", "Tehran")
    with col_button:
        fetch_weather = st.button("Fetch Weather", type="secondary")

    if fetch_weather and city:
        with st.spinner(f"Fetching weather for {city}... (up to ~45s with retries)"):
            try:
                weather = get_weather(city)
                st.success(f"Weather data for {weather.get('city', city)}")

                st.session_state['temp'] = weather.get('temperature') or weather.get('temperature_celsius', 22.0)
                st.session_state['humid'] = weather.get('humidity', 45)
                st.session_state['wind'] = weather.get('wind_kph', 10)
                st.session_state['uv'] = weather.get('uv') or weather.get('uv_index', 3)

                aq = weather.get('air_quality', {}) or {}

                st.session_state['pm25'] = float(aq.get('pm2_5') or 10.0)
                st.session_state['pm10'] = float(aq.get('pm10') or 0.0)

                o3_val = float(aq.get('o3') or 0.0)
                st.session_state['o3'] = min(max(o3_val, 0.0), 0.5)

                no2_val = float(aq.get('no2') or 0.0)
                st.session_state['no2'] = min(max(no2_val, 0.0), 2.0)

                so2_val = float(aq.get('so2') or 0.0)
                st.session_state['so2'] = min(max(so2_val, 0.0), 1.0)

                co_val = float(aq.get('co') or 0.0)
                st.session_state['co'] = min(max(co_val, 0.0), 50.0)

            except Exception as e:
                st.error(f"Could not fetch weather: {e}")
                st.info(
                    "Troubleshooting:\n"
                    "1. Make sure the FastAPI server is running:\n"
                    "   python -m uvicorn api.ed_api:app --host 0.0.0.0 --port 8000\n"
                    "2. Check that ED_API_URL matches your server.\n"
                    "3. Retry if Open-Meteo is temporarily slow."
                )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Weather")
        temp = st.number_input("Temperature (°C)", -50.0, 60.0,
                              value=float(st.session_state.get('temp', 22.0)), step=0.5, key="temp")
        humid = st.slider("Humidity (%)", 0, 100,
                         value=int(st.session_state.get('humid', 45)), key="humid")
        wind = st.slider("Wind (kph)", 0, 100,
                        value=int(st.session_state.get('wind', 10)), key="wind")
        uv = st.slider("UV Index", 0, 15,
                      value=int(st.session_state.get('uv', 3)), key="uv")

    with col2:
        st.markdown("#### Air Quality")
        st.caption("Enter pollutant concentrations (leave 0 if unknown)")
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0,
                               value=float(st.session_state.get('pm25', 10.0)), step=0.1, key="pm25")
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 600.0,
                               value=float(st.session_state.get('pm10', 0.0)), step=0.1, key="pm10")
        o3 = st.number_input("O3 (ppm)", 0.0, 0.5,
                             value=float(st.session_state.get('o3', 0.0)), step=0.001, format="%.3f", key="o3")
        no2 = st.number_input("NO2 (ppm)", 0.0, 2.0,
                              value=float(st.session_state.get('no2', 0.0)), step=0.001, format="%.3f", key="no2")
        so2 = st.number_input("SO2 (ppm)", 0.0, 1.0,
                              value=float(st.session_state.get('so2', 0.0)), step=0.001, format="%.3f", key="so2")
        co = st.number_input("CO (ppm)", 0.0, 50.0,
                             value=float(st.session_state.get('co', 0.0)), step=0.1, key="co")

        cluster_id = st.number_input("Cluster ID (0-5, -1 for auto)", -1, 5, -1, step=1, key="cluster")
        if cluster_id == -1:
            cluster_id = None

    pollutants = {
        "PM2.5": pm25 if pm25 > 0 else None,
        "PM10": pm10 if pm10 > 0 else None,
        "O3": o3 if o3 > 0 else None,
        "NO2": no2 if no2 > 0 else None,
        "SO2": so2 if so2 > 0 else None,
        "CO": co if co > 0 else None,
    }

    aqi_values = {}
    if pollutants["PM2.5"] is not None:
        aqi_values["PM2.5"] = pm25_to_aqi(pollutants["PM2.5"])
    if pollutants["PM10"] is not None:
        aqi_values["PM10"] = pm10_to_aqi(pollutants["PM10"])
    if pollutants["O3"] is not None:
        aqi_values["O3"] = o3_to_aqi(pollutants["O3"])
    if pollutants["NO2"] is not None:
        aqi_values["NO2"] = no2_to_aqi(pollutants["NO2"])
    if pollutants["SO2"] is not None:
        aqi_values["SO2"] = so2_to_aqi(pollutants["SO2"])
    if pollutants["CO"] is not None:
        aqi_values["CO"] = co_to_aqi(pollutants["CO"])

    if aqi_values:
        max_aqi = max(aqi_values.values())
        max_pollutant = max(aqi_values.items(), key=lambda x: x[1])
        epa_index = aqi_to_epa_index(max_aqi)
        epa_label = EPA_LABELS[epa_index]

        penalty = calculate_multi_pollutant_penalty(pollutants)
        high_count = sum(1 for aqi in aqi_values.values() if aqi > 100)

        if high_count >= 3:
            penalty_text = f"+5 pts: {high_count} pollutants with AQI > 100"
        elif high_count == 2:
            penalty_text = f"+2 pts: {high_count} pollutants with AQI > 100"
        else:
            penalty_text = "No multi-pollutant penalty"

        st.info(
            f"Max AQI: {max_aqi:.0f} (from {max_pollutant[0]})  |  "
            f"EPA Index: {epa_index} – {epa_label}"
        )
        st.caption(penalty_text)

        with st.expander("Individual Pollutant AQI Values"):
            for name, aqi in sorted(aqi_values.items(), key=lambda x: x[1], reverse=True):
                epa = aqi_to_epa_index(aqi)
                high = "🚨" if aqi > 100 else ""
                st.write(f"- {name}: {aqi:.0f} (EPA: {epa}) {high}")
    else:
        epa_index = 1
        max_aqi = 0
        st.info("No pollutant data entered. Assuming Good air quality (EPA Index: 1)")
        penalty = 0
        high_count = 0

    anomaly_flag = st.checkbox("Anomaly Override (force 100)", value=False)

    if st.button("Predict Exercise Danger", type="primary", use_container_width=True):
        app_temp = apparent_temp(temp, humid)

        try:
            with st.spinner("Computing ED score..."):
                result = model.predict(
                    temperature_celsius=temp,
                    humidity=humid,
                    wind_kph=wind,
                    uv_index=uv,
                    air_quality_us_epa_index=float(epa_index),
                    air_quality_PM2_5=pm25 if pm25 > 0 else None,
                    air_quality_PM10=pm10 if pm10 > 0 else None,
                    air_quality_Ozone=o3 if o3 > 0 else None,
                    air_quality_Nitrogen_dioxide=no2 if no2 > 0 else None,
                    air_quality_Sulphur_dioxide=so2 if so2 > 0 else None,
                    air_quality_Carbon_Monoxide=co if co > 0 else None,
                    cluster_id=cluster_id,
                    anomaly_flag=anomaly_flag,
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        score = result["ED"]
        category = result["Category"]
        breakdown = result["breakdown"]
        regional_adjustment = result["regional_adjustment"]
        baseline_ed = result["baseline_ed"]
        confidence_range = result.get("confidence_range", "0 - 100")
        safety_floor = result["safety_floor_activated"]

        st.markdown("---")
        st.subheader("Results")

        color = category_color(category)
        category_display = category_clean(category)

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="color: white; margin: 0;">{score:.1f}</h1>
                    <p style="color: white; margin: 0; font-size: 18px;">{category_display}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.metric("Apparent Temp", f"{app_temp:.1f}°C")

        with col3:
            if aqi_values:
                st.metric("Max AQI", f"{max_aqi:.0f}")
            else:
                st.metric("EPA Index", f"{epa_index}")

        with col4:
            st.metric("Confidence", confidence_range)

        st.progress(int(score), text=f"Risk Score: {score:.1f}/100")

        if regional_adjustment != 0:
            adj_text = f"{regional_adjustment:+.2f} points"
            adj_color = "🟢" if regional_adjustment < 0 else "🔴" if regional_adjustment > 0 else "⚪"
            st.info(f"{adj_color} Regional Adjustment: {adj_text} (Cluster {cluster_id if cluster_id is not None else 'N/A'})")

        if safety_floor:
            st.warning("Safety Floor Activated: An extreme component (>70) triggered risk override.")

        st.markdown("#### Score Breakdown")
        cols = st.columns(5)
        breakdown_items = [
            ("Heat", breakdown.get("heat", 0)),
            ("Air", breakdown.get("air", 0)),
            ("UV", breakdown.get("uv", 0)),
            ("Cold", breakdown.get("cold", 0)),
            ("Synergy", breakdown.get("synergy", 0)),
        ]
        for col, (label, value) in zip(cols, breakdown_items):
            with col:
                st.metric(label, f"{value:.1f}")

        if high_count >= 2:
            st.warning(f"Multi-pollutant alert: {high_count} pollutants have AQI > 100. "
                      f"Air quality score adjusted by +{penalty} points.")

        st.markdown("#### Interpretation")
        if anomaly_flag:
            st.error("ANOMALY OVERRIDE: Exercise not advised.")
        elif score >= 75:
            st.error("EXTREME DANGER: Do not exercise outdoors.")
        elif score >= 50:
            st.error("DANGEROUS: Limit outdoor exercise severely.")
        elif score >= 30:
            st.warning("CAUTION: Exercise with care.")
        elif score >= 15:
            st.info("MODERATE SAFE: Generally safe.")
        else:
            st.success("VERY SAFE: Excellent conditions.")

        with st.expander("Raw Prediction Output", expanded=False):
            output = {
                "ED": score,
                "Risk_Level": result["Risk_Level"],
                "Category": category,
                "breakdown": breakdown,
                "baseline_ed": baseline_ed,
                "regional_adjustment": regional_adjustment,
                "safety_floor_activated": safety_floor,
                "confidence_range": confidence_range,
                "EPA_Index": epa_index,
                "Cluster": cluster_id,
                "Model_Type": "Grid-based GNN (1938 nodes)",
            }
            if aqi_values:
                output["AQI_values"] = {k: round(v, 1) for k, v in aqi_values.items()}
                output["max_AQI"] = max_aqi
                output["primary_pollutant"] = max_pollutant[0]
                if high_count >= 2:
                    output["multi_pollutant_penalty"] = penalty
                    output["pollutants_above_100"] = high_count
            st.json(output)

if __name__ == "__main__":
    main()