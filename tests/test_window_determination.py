"""
   Copyright 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import datetime
import unittest

import numpy as np
import pandas as pd

from algo import dynamic_window_determination_user_profile as dwdup

DEFAULT_BOUNDARIES = [datetime.time(hour) for hour in range(0, 24, 3)]


def meter(start, days):
    # 5-minute readings of a meter with a morning and an evening peak that vary from day to day,
    # without randomness so that the expected boundaries stay valid
    index = pd.date_range(start, periods=days * 288, freq="5min")
    hour = np.asarray(index.hour + index.minute / 60)
    day = np.asarray((index - index[0]).days)
    rate = (0.02 + 0.08 * np.exp(-(hour - 7) ** 2 / 2) * (1 + 0.3 * np.sin(day))
            + 0.12 * np.exp(-(hour - 19) ** 2 / 3) * (1 + 0.2 * np.cos(day)))
    return pd.Series(100 + np.cumsum(rate), index=index)


class TestTenMinuteConsumptions(unittest.TestCase):
    def test_takes_the_consumption_of_a_slot_from_its_first_to_its_last_value(self):
        minutes = pd.date_range("2026-03-02", periods=2 * 1440, freq="T")
        frame = dwdup.compute_ten_min_consumptions(pd.Series(np.arange(len(minutes)) * 0.25, index=minutes))
        self.assertEqual((2, 144), frame.shape)
        # nine steps of a minute from the first to the last value of a slot
        self.assertTrue((frame == 1000 * 9 * 0.25).all().all())

    def test_leaves_out_the_days_without_all_ten_minute_slots(self):
        minutes = pd.date_range("2026-03-02 07:23", "2026-03-05 12:00", freq="T")
        frame = dwdup.compute_ten_min_consumptions(pd.Series(np.arange(len(minutes)) * 0.25, index=minutes))
        self.assertEqual([datetime.date(2026, 3, 3), datetime.date(2026, 3, 4)], list(frame.index))


class TestWindowDivision(unittest.TestCase):
    def consumptions(self, series):
        return {name: 1000 * (window.iloc[-1] - window.iloc[0]) if len(window) else None
                for name, window in dwdup.window_division(DEFAULT_BOUNDARIES, series)[0][1]}

    def test_ends_the_last_window_of_a_day_at_the_first_value_of_the_next(self):
        index = pd.date_range("2026-03-02 00:00:30", periods=2 * 288, freq="5min")
        consumptions = self.consumptions(pd.Series(np.arange(len(index)) * 0.25, index=index))
        # as every other window, which ends at the first value of the window after it
        self.assertEqual(consumptions["18:00:00-21:00:00"], consumptions["21:00:00-00:00:00"])

    def test_leaves_the_last_window_empty_when_the_day_has_no_values_in_it(self):
        index = pd.date_range("2026-03-02 00:00:30", periods=2 * 288, freq="5min")
        series = pd.Series(np.arange(len(index)) * 0.25, index=index)
        series = series.drop(series["2026-03-02 21:00":"2026-03-02 23:59"].index)
        self.assertIsNone(self.consumptions(series)["21:00:00-00:00:00"])


def meter_by_day_type(start, days, weekend_peaks):
    # 5-minute readings of a meter with a morning and an evening peak on weekdays and the given peaks on weekends
    index = pd.date_range(start, periods=days * 288, freq="5min")
    hour = np.asarray(index.hour + index.minute / 60)
    weekend = np.asarray(index.dayofweek >= 5)
    weekday_rate = 0.08 * np.exp(-(hour - 7) ** 2 / 2) + 0.12 * np.exp(-(hour - 19) ** 2 / 3)
    weekend_rate = sum(0.1 * np.exp(-(hour - peak) ** 2 / 2) for peak in weekend_peaks)
    return pd.Series(100 + np.cumsum(0.02 + np.where(weekend, weekend_rate, weekday_rate)), index=index)


class TestWindowDetermination(unittest.TestCase):
    def test_determines_the_boundaries_of_weekdays_from_weekdays_only(self):
        other_weekends = meter_by_day_type("2026-03-02", 28, weekend_peaks=[3, 12])
        self.assertEqual(dwdup.window_determination(meter_by_day_type("2026-03-02", 28, weekend_peaks=[10]), DEFAULT_BOUNDARIES, "weekday"),
                         dwdup.window_determination(other_weekends, DEFAULT_BOUNDARIES, "weekday"))

    def test_determines_the_boundaries_of_weekends_from_weekends_only(self):
        history = meter_by_day_type("2026-03-02", 28, weekend_peaks=[3, 12])
        self.assertNotEqual(dwdup.window_determination(history, DEFAULT_BOUNDARIES, "weekday"),
                            dwdup.window_determination(history, DEFAULT_BOUNDARIES, "weekend"))

    def test_leaves_out_a_day_on_which_the_clocks_go_back(self):
        # equal readings every 5 minutes of real time in local time, as the operator keeps them: of the hour
        # that comes twice on October 25, 2026 only the first pass; the weekend before is the only other one
        utc = pd.date_range("2026-10-22 22:00:30", "2026-10-26 23:00", freq="5min", tz="UTC")
        meter = pd.Series(100 + 0.25 * np.arange(len(utc)), index=utc)
        meter = meter[(utc < "2026-10-25 01:00") | (utc >= "2026-10-25 02:00")]
        history = pd.Series(meter.to_numpy(), index=meter.index.tz_convert("Europe/Berlin").tz_localize(None))
        # without the extra hour of that day in a single ten-minute slot the profile is flat
        self.assertEqual(DEFAULT_BOUNDARIES, dwdup.window_determination(history, DEFAULT_BOUNDARIES, "weekend"))

    def test_finds_the_boundaries_of_a_month_with_partial_days_and_a_gap(self):
        history = meter("2026-03-02 07:23", 30)
        history = history.drop(history["2026-03-10 13:00":"2026-03-10 16:00"].index)
        boundaries = dwdup.window_determination(history, DEFAULT_BOUNDARIES)
        # as the determination before its vectorization found them
        self.assertEqual(['00:00:00', '01:35:00', '03:10:00', '05:57:30', '08:45:00', '11:32:30', '14:20:00',
                          '17:13:20', '20:06:40', '23:00:00'], [str(t) for t in boundaries])


if __name__ == '__main__':
    unittest.main()
