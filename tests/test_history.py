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
import tempfile
import unittest

import numpy as np
import pandas as pd

from algo import dynamic_window_determination_user_profile as dwdup
from main import CustomConfig, Operator

CEST = datetime.timezone(datetime.timedelta(hours=2))
DEFAULT_BOUNDARIES = [datetime.time(hour) for hour in range(0, 24, 3)]


def meter(start, days, peaks):
    # 5-minute readings of a meter whose consumption peaks at the given hours of the day
    index = pd.date_range(start, periods=days * 288, freq="5min")
    hour = np.asarray(index.hour + index.minute / 60)
    rate = 0.02 + sum(0.1 * np.exp(-(hour - peak) ** 2 / 2) for peak in peaks)
    return pd.Series(np.cumsum(rate), index=index)


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.operator = self.start()

    def tearDown(self):
        self.dir.cleanup()

    def start(self):
        operator = Operator()
        operator.produce = lambda value: None
        operator.init(None, None, None, None, None, None,
                      config=CustomConfig({"data_path": self.dir.name, "init_phase_length": "2",
                                           "init_phase_level": "d"}))
        return operator

    def test_determines_the_time_windows_from_the_last_90_days(self):
        earlier = meter("2026-01-01", 30, peaks=[3, 12])
        recent = meter("2026-01-31", 91, peaks=[7, 19]) + earlier.iloc[-1]
        history = pd.concat([earlier, recent])
        self.operator.data_history = history
        self.operator.timestamp = history.index[-1]
        self.operator.update_time_window_data()
        last_90_days = dwdup.window_determination(history[history.index[-1] - pd.Timedelta(90, "d"):], DEFAULT_BOUNDARIES, "weekday")
        self.assertNotEqual(dwdup.window_determination(history, DEFAULT_BOUNDARIES, "weekday"), last_90_days)
        self.assertEqual(last_90_days, self.operator.window_boundaries["weekday"])

    def test_a_crash_loses_at_most_the_history_of_the_current_hour(self):
        at = datetime.datetime(2026, 6, 1, 10, 0, 30, tzinfo=CEST)
        while at <= datetime.datetime(2026, 6, 1, 12, 20, 30, tzinfo=CEST):
            self.operator.run({"Consumption": 100}, "", "device", at)
            at += datetime.timedelta(minutes=5)
        restarted = self.start()  # without stop, as after a crash
        self.assertEqual([pd.Timestamp("2026-06-01 10:00:30"), pd.Timestamp("2026-06-01 12:00:30")],
                         [restarted.data_history.index.min(), restarted.data_history.index.max()])


if __name__ == '__main__':
    unittest.main()
