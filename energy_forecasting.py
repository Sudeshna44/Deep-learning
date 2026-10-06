from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

LOOKBACK_STEPS = 96
FORECAST_STEPS = 4
UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)


def prepare_power_series(
    data: pd.DataFrame, *, power_column: str | None = None
) -> pd.Series:
    """Parse active-power readings and resample them to 15-minute means."""
    columns = {str(column).strip().lower(): column for column in data.columns}
    selected_power_column = (
        columns.get(power_column.strip().lower())
        if power_column is not None
        else columns.get("global_active_power")
    )
    if selected_power_column is None:
        raise ValueError(
            "The input must include a Global_active_power or "
            "PowerConsumption_Zone column."
        )

    if "date" in columns and "time" in columns:
        timestamps = pd.to_datetime(
            data[columns["date"]].astype(str).str.strip()
            + " "
            + data[columns["time"]].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )
    else:
        datetime_column = columns.get("datetime") or columns.get("timestamp")
        if datetime_column is None:
            raise ValueError(
                "The input must include Date and Time columns or a "
                "Datetime/Timestamp column."
            )
        timestamps = pd.to_datetime(data[datetime_column], errors="coerce")

    power = pd.to_numeric(
        data[selected_power_column].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    readings = pd.Series(power.to_numpy(), index=timestamps, name="power_kw")
    readings = readings[readings.index.notna() & readings.notna()]
    if readings.empty:
        raise ValueError("No valid timestamped active-power readings were found.")

    readings = readings.sort_index()
    quarter_hour = readings.resample("15min").mean()
    quarter_hour = quarter_hour.interpolate(
        method="time", limit=4, limit_area="inside"
    ).dropna()
    if quarter_hour.empty:
        raise ValueError("No usable 15-minute readings remain after resampling.")
    if (quarter_hour < 0).any():
        raise ValueError("Active-power readings cannot be negative.")
    return quarter_hour.astype(float).rename("power_kw")


def load_uci_series() -> pd.Series:
    """Download and parse the public UCI Individual Household dataset."""
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "The requests package is required to download the UCI dataset."
        ) from exc

    response = requests.get(UCI_DATASET_URL, timeout=60)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        member = next(
            (
                name
                for name in archive.namelist()
                if name.lower().endswith("household_power_consumption.txt")
            ),
            None,
        )
        if member is None:
            raise ValueError("The downloaded UCI archive did not contain its data file.")
        with archive.open(member) as data_file:
            data = pd.read_csv(
                data_file, sep=";", na_values=["?"], low_memory=False
            )
    return prepare_power_series(data)


def daily_energy_kwh(power_series: pd.Series) -> pd.Series:
    """Estimate daily energy as the sum of quarter-hour mean kW × 0.25 h."""
    return power_series.resample("D").sum(min_count=1).mul(0.25).rename("energy_kwh")


def train_lstm(
    power_series: pd.Series,
    *,
    epochs: int = 8,
    batch_size: int = 64,
) -> tuple[object, float, float, dict[str, float]]:
    """Train a chronological train/validation LSTM and report kW validation errors."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for LSTM training. Install dependencies from "
            "requirements.txt."
        ) from exc

    values = power_series.to_numpy(dtype=np.float32)
    window_size = LOOKBACK_STEPS + FORECAST_STEPS
    split = int(len(values) * 0.8)
    if split <= window_size or len(values) - split < FORECAST_STEPS:
        raise ValueError(
            "Training history is too short for a chronological training and "
            "validation split."
        )

    center = float(values[:split].mean())
    scale = float(values[:split].std())
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Training readings must have non-zero variation.")
    normalized = (values - center) / scale

    windows = np.lib.stride_tricks.sliding_window_view(normalized, window_size)
    starts = np.arange(len(windows))
    train_mask = starts + window_size <= split
    validation_mask = starts + LOOKBACK_STEPS >= split
    train_windows = windows[train_mask]
    validation_windows = windows[validation_mask]
    if not len(train_windows) or not len(validation_windows):
        raise ValueError("Not enough readings to create training and validation windows.")

    x_train = train_windows[:, :LOOKBACK_STEPS, np.newaxis].copy()
    y_train = train_windows[:, LOOKBACK_STEPS:].copy()
    x_validation = validation_windows[:, :LOOKBACK_STEPS, np.newaxis].copy()
    y_validation = validation_windows[:, LOOKBACK_STEPS:].copy()

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK_STEPS, 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(FORECAST_STEPS),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
    )

    normalized_predictions = model.predict(x_validation, verbose=0)
    errors_kw = (normalized_predictions - y_validation) * scale
    metrics = {
        "mae_kw": float(np.mean(np.abs(errors_kw))),
        "rmse_kw": float(np.sqrt(np.mean(np.square(errors_kw)))),
    }
    return model, center, scale, metrics


def predict(
    model: object, recent_readings: np.ndarray, center: float, scale: float
) -> np.ndarray:
    """Return the next four 15-minute forecasts in kW."""
    recent = np.asarray(recent_readings, dtype=np.float32)
    if recent.shape != (LOOKBACK_STEPS,):
        raise ValueError(f"Expected exactly {LOOKBACK_STEPS} recent readings.")
    normalized = ((recent - center) / scale).reshape(1, LOOKBACK_STEPS, 1)
    predictions = np.asarray(model.predict(normalized, verbose=0)).reshape(-1)
    if predictions.size != FORECAST_STEPS:
        raise ValueError(f"The model must return {FORECAST_STEPS} forecast values.")
    return np.maximum(predictions * scale + center, 0.0)
