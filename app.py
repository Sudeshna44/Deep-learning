from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from energy_forecasting import (
    LOOKBACK_STEPS,
    daily_energy_kwh,
    load_uci_series,
    prepare_power_series,
    predict,
    train_lstm,
)


st.set_page_config(page_title="Smart Home Energy Forecast", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #172b3a;
        --muted: #617581;
        --green: #087f69;
        --green-dark: #066653;
        --mint: #e8f5ef;
        --line: #e5ece8;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(209, 239, 222, .5), transparent 32rem),
            #f7faf8;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: 'Manrope', 'DM Sans', sans-serif;
        color: var(--ink);
        letter-spacing: -.035em;
    }
    h2 { font-size: 1.45rem !important; }
    [data-testid="stCaptionContainer"], .stMarkdown p {
        color: var(--muted);
    }
    [data-testid="stSidebar"] {
        background: #f0f6f2;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: .92rem;
    }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 18px rgba(23, 43, 58, .035);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: .88rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stButton > button[kind="primary"] {
        border: 0;
        border-radius: 11px;
        background: var(--green);
        color: white;
        font-weight: 700;
        padding: .65rem 1rem;
        transition: background .15s ease, transform .15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--green-dark);
        transform: translateY(-1px);
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, .75);
        border-radius: 12px;
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem;
        margin: .2rem 0 1.5rem;
        border: 1px solid rgba(8, 127, 105, .12);
        border-radius: 24px;
        background: linear-gradient(115deg, #e8f5ef 0%, #f2f8f3 55%, #e9f2ef 100%);
    }
    .hero::after {
        content: '⚡';
        position: absolute;
        right: 5%;
        top: -2.2rem;
        color: rgba(8, 127, 105, .09);
        font-size: 11rem;
        transform: rotate(12deg);
    }
    .hero-kicker {
        position: relative;
        z-index: 1;
        margin-bottom: .55rem;
        color: var(--green);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 800;
        line-height: 1.12;
    }
    .hero p {
        position: relative;
        z-index: 1;
        max-width: 660px;
        margin: .75rem 0 0;
        color: #526b70;
        font-size: 1rem;
        line-height: 1.65;
    }
    .section-note {
        margin-top: -.55rem;
        margin-bottom: 1rem;
        color: var(--muted);
    }
    .welcome-card {
        height: 100%;
        padding: 1.25rem 1.35rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: #fff;
    }
    .welcome-card strong {
        display: block;
        margin-bottom: .35rem;
        color: var(--ink);
        font-size: 1rem;
    }
    .welcome-card span {
        color: var(--muted);
        font-size: .91rem;
        line-height: 1.55;
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 3rem; }
        .hero { padding: 1.45rem 1.3rem; border-radius: 18px; }
        .hero::after { right: -1.2rem; top: 0; font-size: 6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_project_csv(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


def show_dashboard(series: pd.Series) -> None:
    st.subheader("Energy overview")
    st.markdown(
        '<p class="section-note">A quick snapshot of the latest readings in your selected dataset.</p>',
        unsafe_allow_html=True,
    )
    latest_time = series.index[-1]
    current_power = float(series.iloc[-1])
    today = series[series.index.date == latest_time.date()]
    today_energy = float(daily_energy_kwh(series).get(pd.Timestamp(latest_time.date()), 0.0))
    today_peak = float(today.max()) if not today.empty else current_power

    with st.container(border=True):
        metric_columns = st.columns(3)
        metric_columns[0].metric("Current power", f"{current_power:.2f} kW")
        metric_columns[1].metric("Energy today", f"{today_energy:.2f} kWh")
        metric_columns[2].metric("Today's peak", f"{today_peak:.2f} kW")

    threshold_default = round(max(4.5, current_power * 1.15), 1)
    threshold_max = round(max(10.0, threshold_default * 2), 1)
    threshold = st.slider(
        "High-consumption alert threshold (kW)",
        min_value=0.1,
        max_value=threshold_max,
        value=threshold_default,
        step=max(0.1, round(threshold_max / 100, 1)),
        key="alert_threshold",
    )
    if current_power >= threshold:
        st.warning(
            f"Current power ({current_power:.2f} kW) is at or above your "
            f"{threshold:.1f} kW alert threshold."
        )
        st.info(
            "Energy-saving idea: check high-draw appliances such as electric "
            "heating, water heating, or cooking equipment, and switch off "
            "devices that are not needed."
        )
    else:
        st.success("Current consumption is below your alert threshold.")

    st.subheader("Train a 15-minute-to-1-hour forecast")
    st.caption(
        "The LSTM uses the previous 24 hours of 15-minute average active-power "
        "readings to predict the next four 15-minute intervals."
    )
    with st.container(border=True):
        settings_columns = st.columns([1.2, 1, 1.1])
        days = settings_columns[0].select_slider(
            "Training history",
            options=[7, 14, 30, 60, 90],
            value=30,
            format_func=lambda value: f"{value} days",
        )
        epochs = settings_columns[1].select_slider(
            "Training effort",
            options=[3, 5, 8, 12, 16, 20],
            value=8,
            format_func=lambda value: f"{value} epochs",
        )
        settings_columns[2].markdown(
            "<br>", unsafe_allow_html=True
        )
        train_clicked = settings_columns[2].button(
            "⚡  Generate forecast", type="primary", use_container_width=True
        )

    if train_clicked:
        training_series = series.last(f"{days}D")
        minimum_points = LOOKBACK_STEPS + 4 + 20
        if len(training_series) < minimum_points:
            st.error(
                f"Not enough usable readings to train and validate the model. "
                f"At least {minimum_points} 15-minute readings are required."
            )
        else:
            try:
                with st.spinner("Training the LSTM; this can take a little while..."):
                    model, center, scale, metrics = train_lstm(
                        training_series, epochs=epochs
                    )
                    forecast = predict(
                        model,
                        training_series.iloc[-LOOKBACK_STEPS:].to_numpy(),
                        center,
                        scale,
                    )
                st.session_state["forecast"] = forecast
                st.session_state["forecast_metrics"] = metrics
                st.session_state["forecast_origin"] = training_series.index[-1]
                st.success("Forecast is ready.")
            except (ImportError, RuntimeError, ValueError) as exc:
                st.error(f"Could not train the LSTM: {exc}")

    forecast = st.session_state.get("forecast")
    forecast_origin = st.session_state.get("forecast_origin")
    if forecast is not None and forecast_origin == series.index[-1]:
        st.subheader("Predicted consumption")
        forecast_times = [
            forecast_origin + pd.Timedelta(minutes=15 * step)
            for step in range(1, len(forecast) + 1)
        ]
        forecast_columns = st.columns(len(forecast))
        for step, (column, value) in enumerate(zip(forecast_columns, forecast), start=1):
            column.metric(f"Next {step * 15} min", f"{value:.2f} kW")

        metrics = st.session_state.get("forecast_metrics", {})
        st.caption(
            f"Validation MAE: {metrics.get('mae_kw', float('nan')):.3f} kW · "
            f"RMSE: {metrics.get('rmse_kw', float('nan')):.3f} kW"
        )

        if max(forecast) >= threshold:
            st.warning(
                "The forecast reaches your high-consumption threshold. "
                "Consider postponing flexible high-power appliance use."
            )

        chart = go.Figure()
        recent = series.last("24h")
        chart.add_trace(
            go.Scatter(x=recent.index, y=recent, mode="lines", name="Observed")
        )
        chart.add_trace(
            go.Scatter(
                x=forecast_times,
                y=forecast,
                mode="lines+markers",
                name="LSTM forecast",
                line={"dash": "dash"},
            )
        )
        chart.update_layout(
            title="Recent and forecast active power",
            xaxis_title="Time",
            yaxis_title="Active power (kW)",
            hovermode="x unified",
            template="plotly_white",
            height=360,
            margin={"l": 10, "r": 10, "t": 55, "b": 10},
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
        st.plotly_chart(chart, use_container_width=True)
    elif forecast is not None:
        st.caption("The readings have changed; train again to refresh the forecast.")

    st.subheader("Daily energy use")
    energy_by_day = daily_energy_kwh(series).tail(14)
    daily_chart = go.Figure(
        go.Bar(x=energy_by_day.index, y=energy_by_day.values, name="Energy")
    )
    daily_chart.update_layout(
        xaxis_title="Day",
        yaxis_title="Energy (kWh)",
        title="Daily energy · last 14 days",
        template="plotly_white",
        height=320,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    st.plotly_chart(daily_chart, use_container_width=True)


st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Smart home · Energy insights</div>
        <h1>Know your energy.<br>Plan what comes next.</h1>
        <p>See energy usage at a glance and use an LSTM forecast to understand
        how power demand may change over the next hour.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📂 Your data")
    st.caption("Choose a dataset to explore.")
    local_dataset = Path(__file__).resolve().with_name("powerconsumption.csv")
    sources = []
    if local_dataset.is_file():
        sources.append("Project CSV · powerconsumption.csv")
    sources.extend(["Upload CSV or TXT", "UCI household sample"])
    selected_source = st.selectbox("Dataset source", sources)
    series = None
    dataset_identity = None

    if selected_source == "Project CSV · powerconsumption.csv":
        try:
            raw_data = load_project_csv(
                str(local_dataset), local_dataset.stat().st_mtime_ns
            )
            power_columns = [
                str(column)
                for column in raw_data.columns
                if str(column).strip().lower() == "global_active_power"
                or str(column).strip().lower().startswith("powerconsumption_zone")
            ]
            if not power_columns:
                raise ValueError(
                    "No Global_active_power or PowerConsumption_Zone columns "
                    "were found in powerconsumption.csv."
                )
            selected_power_column = (
                st.selectbox(
                    "Power series",
                    power_columns,
                    format_func=lambda column: (
                        column.replace("PowerConsumption_", "").replace("_", " ")
                    ),
                    key="project_power_column",
                )
                if len(power_columns) > 1
                else power_columns[0]
            )
            series = prepare_power_series(
                raw_data, power_column=selected_power_column
            )
            dataset_identity = f"project:{selected_power_column}"
            st.caption(
                f"Using {selected_power_column}; timestamps are resampled to 15-minute intervals."
            )
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            st.error(f"Could not load the project CSV: {exc}")

    elif selected_source == "Upload CSV or TXT":
        uploaded_file = st.file_uploader(
            "Upload consumption data",
            type=["csv", "txt"],
            help=(
                "Supports UCI Date/Time and Global_active_power columns, "
                "Datetime/Timestamp columns, and PowerConsumption_Zone columns."
            ),
        ),
        if uploaded_file is not None:
            try:
                uploaded_bytes = uploaded_file.getvalue()
                if uploaded_file.name.lower().endswith(".txt"):
                    raw_data = pd.read_csv(
                        BytesIO(uploaded_bytes),
                        sep=";",
                        na_values=["?"],
                        low_memory=False,
                    )
                else:
                    raw_data = pd.read_csv(
                        BytesIO(uploaded_bytes),
                        sep=None,
                        engine="python",
                        na_values=["?"],
                    )
                power_columns = [
                    str(column)
                    for column in raw_data.columns
                    if str(column).strip().lower() == "global_active_power"
                    or str(column).strip().lower().startswith(
                        "powerconsumption_zone"
                    )
                ]
                selected_power_column = (
                    st.selectbox(
                        "Power series",
                        power_columns,
                        format_func=lambda column: (
                            column.replace("PowerConsumption_", "").replace("_", " ")
                        ),
                        key="uploaded_power_column",
                    )
                    if len(power_columns) > 1
                    else power_columns[0] if power_columns else None
                )
                series = prepare_power_series(
                    raw_data, power_column=selected_power_column
                )
                dataset_identity = (
                    f"upload:{uploaded_file.name}:{len(uploaded_bytes)}:"
                    f"{selected_power_column}"
                )
                st.caption(
                    f"Using {selected_power_column}; timestamps are resampled to 15-minute intervals."
                )
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                st.error(f"Could not read the uploaded file: {exc}")

    elif st.button("Download and load UCI sample"):
        try:
            with st.spinner("Downloading and preparing the UCI dataset..."):
                series = load_uci_series()
            dataset_identity = "uci-household-sample"
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Could not load the UCI dataset: {exc}")
    elif selected_source == "UCI household sample":
        series = st.session_state.get("power_series")
        if st.session_state.get("dataset_identity") != "uci-household-sample":
            series = None
        else:
            dataset_identity = "uci-household-sample"

if series is not None and dataset_identity is not None:
    if st.session_state.get("dataset_identity") != dataset_identity:
        st.session_state.pop("forecast", None)
        st.session_state.pop("alert_threshold", None)
        st.session_state["dataset_identity"] = dataset_identity
    st.session_state["power_series"] = series

if series is None:
    st.subheader("Get started")
    st.markdown(
        '<p class="section-note">Your project CSV is loaded automatically when available. You can also upload readings or use the UCI sample.</p>',
        unsafe_allow_html=True,
    )
    start_columns = st.columns(3)
    start_columns[0].markdown(
        '<div class="welcome-card"><strong>01 · Choose your data</strong>'
        '<span>Use the project CSV, upload your own readings, or choose the UCI sample.</span></div>',
        unsafe_allow_html=True,
    )
    start_columns[1].markdown(
        '<div class="welcome-card"><strong>02 · Explore usage</strong>'
        '<span>Review current power, daily energy, and recent peak consumption.</span></div>',
        unsafe_allow_html=True,
    )
    start_columns[2].markdown(
        '<div class="welcome-card"><strong>03 · Forecast ahead</strong>'
        '<span>Train an LSTM on the previous 24 hours to predict the next four intervals.</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="margin-top:1.25rem;padding:1rem 1.2rem;border:1px solid #e5ece8;
        border-radius:14px;background:#f0f6f2;color:#526b70">
        <strong style="color:#172b3a">Supported data</strong><br>
        UCI-style <code>Date</code>, <code>Time</code>, <code>Global_active_power</code>
        columns or <code>Datetime</code>/<code>Timestamp</code> with
        <code>Global_active_power</code>. Zone datasets with
        <code>PowerConsumption_Zone1</code>, <code>Zone2</code>, and
        <code>Zone3</code> are supported too.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    show_dashboard(series)
