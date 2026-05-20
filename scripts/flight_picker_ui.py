import os
import json
from datetime import time
import joblib
import streamlit as st
import pandas as pd
import numpy as np

DATA_PATH = "../data/raw/Airlines.csv"
AIRPORT_CODES_PATH = os.path.join(os.path.dirname(__file__), "airport_codes.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_pipeline.joblib")

DAY_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
             5: "Friday", 6: "Saturday", 7: "Sunday"}

AIRLINE_NAMES = {
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "CO": "Continental Airlines",
    "DL": "Delta Air Lines",
    "EV": "ExpressJet Airlines",
    "F9": "Frontier Airlines",
    "FL": "AirTran Airways",
    "HA": "Hawaiian Airlines",
    "MQ": "American Eagle Airlines",
    "OH": "PSA Airlines",
    "OO": "SkyWest Airlines",
    "UA": "United Airlines",
    "US": "US Airways",
    "WN": "Southwest Airlines",
    "XE": "ExpressJet Airlines (XE)",
    "YV": "Mesa Airlines",
}

FEATURE_COLS = ["Airline", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length"]


def fmt_time(minutes):
    h, m = int(minutes) // 60, int(minutes) % 60
    period = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {period}"


def fmt_length(minutes):
    minutes = int(minutes)
    if minutes < 60:
        return f"{minutes} minutes"
    h, m = minutes // 60, minutes % 60
    if m == 0:
        return f"{h} hours" if h > 1 else "1 hour"
    h_label = "hour" if h == 1 else "hours"
    return f"{h} {h_label} {m} minutes"


@st.cache_data
def load_airport_names():
    with open(AIRPORT_CODES_PATH) as f:
        return json.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["DayName"] = df["DayOfWeek"].map(DAY_NAMES)
    return df


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_flight(model, row):
    features = pd.DataFrame([{col: row[col] for col in FEATURE_COLS}])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return int(prediction), float(proba[1])


def main():
    st.set_page_config(page_title="Flight Delay Predictor", layout="wide")
    st.title("Flight Delay Predictor")
    st.write("Filter and select a flight to view the model's delay prediction.")

    df = load_data()
    model = load_model()
    airport_names = load_airport_names()

    if model is None:
        st.error(
            "No trained model found. Run `python xgboost_pipeline.py` first to train and save the model."
        )

    # --- Filters ---
    st.subheader("Filters")

    # Row 1: Airline, Departure Airport, Arrival Airport, Day of Week
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        airline_codes = sorted(df["Airline"].unique())
        airline_options = [AIRLINE_NAMES.get(c, c) for c in airline_codes]
        airline_name_to_code = {AIRLINE_NAMES.get(c, c): c for c in airline_codes}
        selected_airline_names = st.multiselect("Airline", airline_options, default=[])
        selected_airlines = [airline_name_to_code[n] for n in selected_airline_names]
    with f2:
        if selected_airlines:
            from_pool = df[df["Airline"].isin(selected_airlines)]
        else:
            from_pool = df
        from_codes = sorted(from_pool["AirportFrom"].unique())
        from_options = [f"{airport_names.get(c, c)} ({c})" for c in from_codes]
        from_name_to_code = {f"{airport_names.get(c, c)} ({c})": c for c in from_codes}
        selected_from_names = st.multiselect("Departure Airport", from_options, default=[])
        selected_from = [from_name_to_code[n] for n in selected_from_names]
    with f3:
        to_pool = from_pool
        if selected_from:
            to_pool = to_pool[to_pool["AirportFrom"].isin(selected_from)]
        to_codes = sorted(to_pool["AirportTo"].unique())
        to_options = [f"{airport_names.get(c, c)} ({c})" for c in to_codes]
        to_name_to_code = {f"{airport_names.get(c, c)} ({c})": c for c in to_codes}
        selected_to_names = st.multiselect("Arrival Airport", to_options, default=[])
        selected_to = [to_name_to_code[n] for n in selected_to_names]
    with f4:
        days = list(DAY_NAMES.values())
        selected_days = st.multiselect("Day of Week", days, default=[])

    # Row 2: Delay Status, Departure Time, Flight Length
    f5, f6, f7 = st.columns(3)
    with f5:
        delay_filter = st.radio("Delay Status", ["All", "Delayed", "On Time"], horizontal=True)
    with f6:
        time_min, time_max = int(df["Time"].min()), int(df["Time"].max())
        t_min = time(time_min // 60, time_min % 60)
        t_max = time(time_max // 60, time_max % 60)
        selected_time_range = st.slider(
            "Departure Time",
            min_value=t_min, max_value=t_max, value=(t_min, t_max),
            format="h:mm a",
        )
        selected_time = (
            selected_time_range[0].hour * 60 + selected_time_range[0].minute,
            selected_time_range[1].hour * 60 + selected_time_range[1].minute,
        )
    with f7:
        length_min, length_max = int(df["Length"].min()), int(df["Length"].max())
        selected_length = st.slider(
            "Flight Length (minutes)",
            length_min, length_max, (length_min, length_max),
        )

    # --- Apply filters ---
    filtered = df.copy()

    if selected_airlines:
        filtered = filtered[filtered["Airline"].isin(selected_airlines)]
    if selected_from:
        filtered = filtered[filtered["AirportFrom"].isin(selected_from)]
    if selected_to:
        filtered = filtered[filtered["AirportTo"].isin(selected_to)]
    if selected_days:
        day_nums = [k for k, v in DAY_NAMES.items() if v in selected_days]
        filtered = filtered[filtered["DayOfWeek"].isin(day_nums)]
    if delay_filter == "Delayed":
        filtered = filtered[filtered["Delay"] == 1]
    elif delay_filter == "On Time":
        filtered = filtered[filtered["Delay"] == 0]

    filtered = filtered[
        (filtered["Time"] >= selected_time[0]) & (filtered["Time"] <= selected_time[1])
    ]
    filtered = filtered[
        (filtered["Length"] >= selected_length[0]) & (filtered["Length"] <= selected_length[1])
    ]

    # --- Matching Flights ---
    st.subheader(f"Matching Flights: {len(filtered):,}")

    if filtered.empty:
        st.warning("No flights match the current filters.")
    else:
        display_cols = ["Airline", "Flight", "AirportFrom", "AirportTo",
                        "DayName", "Time", "Length", "Delay"]
        display_df = filtered[display_cols].copy()
        display_df["AirportFrom"] = display_df["AirportFrom"].map(airport_names).fillna(display_df["AirportFrom"])
        display_df["AirportTo"] = display_df["AirportTo"].map(airport_names).fillna(display_df["AirportTo"])
        display_df["Time"] = display_df["Time"].apply(fmt_time)
        display_df["Length"] = display_df["Length"].apply(fmt_length)

        renamed_df = display_df.rename(columns={"DayName": "Day", "AirportFrom": "From", "AirportTo": "To"})

        if st.session_state.get("selected_flight_idx") is not None:
            idx = st.session_state["selected_flight_idx"]
            st.dataframe(
                renamed_df.iloc[[idx]],
                use_container_width=True,
                hide_index=True,
            )
            if st.button("Pick Another Flight"):
                st.session_state["selected_flight_idx"] = None
                st.rerun()
            row = filtered.iloc[idx]
        else:
            event = st.dataframe(
                renamed_df,
                use_container_width=True,
                height=400,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )

            selected_rows = event.selection.rows
            if not selected_rows:
                st.info("Click on a row above to view flight details.")
                return

            st.session_state["selected_flight_idx"] = selected_rows[0]
            row = filtered.iloc[selected_rows[0]]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Airline", AIRLINE_NAMES.get(row["Airline"], row["Airline"]))
            st.metric("Flight #", row["Flight"])
        with col2:
            st.metric("From", airport_names.get(row["AirportFrom"], row["AirportFrom"]))
            st.metric("To", airport_names.get(row["AirportTo"], row["AirportTo"]))
        with col3:
            st.metric("Day", DAY_NAMES.get(row["DayOfWeek"], row["DayOfWeek"]))
            st.metric("Departure", fmt_time(row["Time"]))

        actual_status = "Delayed" if row["Delay"] == 1 else "On Time"
        actual_color = "#ff4b4b" if row["Delay"] == 1 else "#21c354"
        col_len, col_status = st.columns(2)
        with col_len:
            st.metric("Flight Length", fmt_length(row["Length"]))
        with col_status:
            st.markdown(
                f'<div style="background-color:{actual_color};padding:12px 16px;border-radius:8px;color:white;">'
                f'<p style="margin:0;font-size:0.85rem;opacity:0.85;">Actual Status</p>'
                f'<p style="margin:0;font-size:1.5rem;font-weight:600;">{actual_status}</p></div>',
                unsafe_allow_html=True,
            )

        # --- Predicted ---
        st.divider()

        if model is not None:
            pred_label, pred_proba = predict_flight(model, row)
            pred_status = "Delayed" if pred_label == 1 else "On Time"
            pred_color = "#ff4b4b" if pred_label == 1 else "#21c354"
            correct = pred_label == row["Delay"]

            col_pred, col_conf = st.columns(2)
            with col_pred:
                st.markdown(
                    f'<div style="background-color:{pred_color};padding:12px 16px;border-radius:8px;color:white;">'
                    f'<p style="margin:0;font-size:0.85rem;opacity:0.85;">Model Prediction</p>'
                    f'<p style="margin:0;font-size:1.5rem;font-weight:600;">{pred_status}</p></div>',
                    unsafe_allow_html=True,
                )
            with col_conf:
                st.metric("Delay Probability", f"{pred_proba:.1%}")

            if correct:
                st.success("The model's prediction matches the actual outcome.")
            else:
                st.error("The model's prediction differs from the actual outcome.")
        else:
            st.info("Train the model to see predictions here.")

        st.session_state["selected_flight"] = row.to_dict()


if __name__ == "__main__":
    main()
