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

from unittest import mock

import numpy as np
import pandas as pd
from operator_lib.util.config import MissingConfigValueError
from operator_lib.util.helpers import HistoryNeedsRayError

from algo import dynamic_window_determination_user_profile as dwdup
from fake_timescale import Timescale
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
        self.timescale = Timescale()

    def tearDown(self):
        self.dir.cleanup()

    def start(self, reader=None):
        # reads the history from self.timescale unless another reader is given
        operator = Operator()
        operator.produce = lambda value: None
        operator.data_path = self.dir.name
        with reader or self.timescale.patched() as self.read:
            operator.init(None, None, None, None, None, None,
                          config=CustomConfig({"init_phase_length": "2",
                                               "init_phase_level": "d"}))
        return operator

    def test_determines_the_time_windows_from_the_last_90_days(self):
        operator = self.start()
        earlier = meter("2026-01-01", 30, peaks=[3, 12])
        recent = meter("2026-01-31", 91, peaks=[7, 19]) + earlier.iloc[-1]
        history = pd.concat([earlier, recent])
        operator.data_history = history
        operator.timestamp = history.index[-1]
        operator.update_time_window_data()
        last_90_days = dwdup.window_determination(history[history.index[-1] - pd.Timedelta(90, "d"):], DEFAULT_BOUNDARIES, "weekday")
        self.assertNotEqual(dwdup.window_determination(history, DEFAULT_BOUNDARIES, "weekday"), last_90_days)
        self.assertEqual(last_90_days, operator.window_boundaries["weekday"])

    def test_reads_the_last_90_days_from_timescale_in_german_time(self):
        self.timescale.write(datetime.datetime(2026, 6, 1, 10, 0, 30, tzinfo=CEST), 100.0)
        self.timescale.write(datetime.datetime(2026, 6, 1, 10, 5, 30, tzinfo=CEST), 100.25)
        operator = self.start()
        self.read.assert_called_once_with(pd.Timedelta(90, "d"))
        self.assertEqual({pd.Timestamp("2026-06-01 10:00:30"): 100.0, pd.Timestamp("2026-06-01 10:05:30"): 100.25},
                         operator.data_history.to_dict())

    def test_leaves_out_missing_values_and_values_no_later_than_those_before_them_as_live(self):
        # On October 25, 2026, the clocks go back from 03:00 to 02:00, so the hour from 02:00 comes twice.
        for utc, consumption in [("2026-10-25 00:30", 100.0),  # 02:30 summer time
                                 ("2026-10-25 00:30", 100.5),  # the same time again
                                 ("2026-10-25 00:35", None),
                                 ("2026-10-25 01:30", 101.0),  # 02:30 winter time
                                 ("2026-10-25 02:30", 102.0)]:
            self.timescale.write(pd.Timestamp(utc, tz="UTC"), consumption)
        self.assertEqual({pd.Timestamp("2026-10-25 02:30"): 100.0, pd.Timestamp("2026-10-25 03:30"): 102.0},
                         self.start().data_history.to_dict())

    def test_starts_without_history_where_timescale_does_not_have_it(self):
        for error in [HistoryNeedsRayError("the history of input topic t (d) is only in Kafka"),
                      MissingConfigValueError("cannot read history for topic t")]:
            with self.subTest(error=type(error).__name__):
                operator = self.start(mock.patch("main.provide_historic_frames_local", side_effect=error))
                self.assertTrue(operator.data_history.empty)
                result = operator.run({"Consumption": 100}, "", "device", datetime.datetime(2026, 6, 1, 10, 0, 30, tzinfo=CEST))
                self.assertIn("2026-06-03T10:00:30", result["initial_phase"])

    def test_starts_without_history_for_more_than_one_device(self):
        frame = pd.DataFrame({"time": [pd.Timestamp("2026-06-01 08:00:30")] * 2, "Consumption": [100.0, 200.0],
                              "device_id": ["a", "b"]})
        operator = self.start(mock.patch("main.provide_historic_frames_local", return_value=[frame]))
        self.assertTrue(operator.data_history.empty)

    def test_does_not_start_when_timescale_cannot_be_read(self):
        with self.assertRaises(ConnectionError):
            self.start(mock.patch("main.provide_historic_frames_local", side_effect=ConnectionError("no connection")))


if __name__ == '__main__':
    unittest.main()
