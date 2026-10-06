# Smart Home Energy Forecasting

A Streamlit learning project that resamples power readings to 15-minute
intervals and trains an LSTM to forecast active power 15, 30, 45, and
60 minutes ahead from the preceding 24 hours of readings.

## Features

- Automatically load `powerconsumption.csv` from the project folder, select
  one of its Zone 1/2/3 power series, or upload a compatible CSV/TXT file.
- Optionally download the public UCI household dataset.
- View latest active power, today's estimated energy, and today's peak.
- Train a small Keras LSTM on recent history and view a four-step forecast.
- Inspect chronological validation MAE/RMSE and recent/daily-use charts.
- Set a high-consumption threshold and see a basic alert and energy-saving tip.

The UCI dataset's `Global_active_power` field is measured in kilowatts. The
project CSV's `PowerConsumption_Zone1`, `PowerConsumption_Zone2`, and
`PowerConsumption_Zone3` columns are also supported. Energy is estimated by
multiplying each 15-minute mean-power reading by 0.25 hours. Short internal
data gaps are interpolated; longer gaps are left out of model windows.

## Run locally

Use Python 3.10-3.12 in a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

To test the data preparation and energy calculations:

```powershell
python -m unittest discover -s tests
```

## Uploaded data format

The app accepts the original UCI semicolon-separated format with `Date`,
`Time`, and `Global_active_power` columns; CSV files with a `Datetime` or
`Timestamp` column and `Global_active_power`; or `Datetime` and
`PowerConsumption_Zone1/2/3` columns. Unknown or invalid power values are
ignored during resampling.

## Dataset

UCI Machine Learning Repository, [Individual Household Electric Power
Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

## Notes

Forecasts are educational estimates, not meter readings or utility-grade
predictions. Model quality depends on data quality and the selected training
period. TensorFlow installation and training can be resource-intensive on
machines without a supported CPU/GPU build.
