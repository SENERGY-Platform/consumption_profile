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


class TestWindowDetermination(unittest.TestCase):
    def test_finds_the_boundaries_of_a_month_with_partial_days_and_a_gap(self):
        history = meter("2026-03-02 07:23", 30)
        history = history.drop(history["2026-03-10 13:00":"2026-03-10 16:00"].index)
        boundaries = dwdup.window_determination(history, DEFAULT_BOUNDARIES)
        # as the determination before its vectorization found them
        self.assertEqual(['00:00:00', '01:35:00', '03:10:00', '05:57:30', '08:45:00', '11:32:30', '14:20:00',
                          '17:13:20', '20:06:40', '23:00:00'], [str(t) for t in boundaries])


if __name__ == '__main__':
    unittest.main()
