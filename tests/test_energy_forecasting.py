import unittest

import pandas as pd

from energy_forecasting import daily_energy_kwh, prepare_power_series


class PreparePowerSeriesTests(unittest.TestCase):
    def test_combines_uci_date_and_time_and_resamples_to_quarter_hours(self):
        data = pd.DataFrame(
            {
                "Date": ["1/1/2007"] * 30,
                "Time": [f"00:{minute:02d}:00" for minute in range(30)],
                "Global_active_power": [str(value) for value in range(30)],
            }
        )

        result = prepare_power_series(data)

        self.assertEqual(len(result), 2)
        self.assertEqual(result.index[0], pd.Timestamp("2007-01-01 00:00"))
        self.assertAlmostEqual(result.iloc[0], 7.0)
        self.assertAlmostEqual(result.iloc[1], 22.0)

    def test_interpolates_short_internal_gaps(self):
        data = pd.DataFrame(
            {
                "Datetime": [
                    "2024-01-01 00:00",
                    "2024-01-01 00:30",
                ],
                "Global_active_power": [2.0, 4.0],
            }
        )

        result = prepare_power_series(data)

        self.assertEqual(result.tolist(), [2.0, 3.0, 4.0])

    def test_uses_selected_power_consumption_zone(self):
        data = pd.DataFrame(
            {
                "Datetime": [
                    "2024-01-01 00:00",
                    "2024-01-01 00:10",
                    "2024-01-01 00:20",
                    "2024-01-01 00:30",
                ],
                "PowerConsumption_Zone1": [10.0, 20.0, 30.0, 40.0],
                "PowerConsumption_Zone2": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = prepare_power_series(
            data, power_column="PowerConsumption_Zone2"
        )

        self.assertEqual(result.tolist(), [1.5, 3.0, 4.0])

    def test_rejects_missing_power_column(self):
        with self.assertRaisesRegex(ValueError, "Global_active_power"):
            prepare_power_series(
                pd.DataFrame({"Datetime": ["2024-01-01"], "Power": [1.0]})
            )


class DailyEnergyTests(unittest.TestCase):
    def test_converts_quarter_hour_mean_power_to_daily_kwh(self):
        index = pd.date_range("2024-01-01", periods=4, freq="15min")
        power = pd.Series([2.0, 2.0, 2.0, 2.0], index=index)

        result = daily_energy_kwh(power)

        self.assertAlmostEqual(result.iloc[0], 2.0)


if __name__ == "__main__":
    unittest.main()
