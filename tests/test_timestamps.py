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

from main import CustomConfig, Operator


class TestOutputTimestamp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.operator = Operator()
        self.operator.produce = lambda value: None
        self.operator.init(None, None, None, None, None, None,
                           config=CustomConfig({"data_path": self.dir.name, "init_phase_length": "2",
                                                "init_phase_level": "d"}))

    def tearDown(self):
        self.dir.cleanup()

    def results(self, start, steps):
        out = []
        for step in range(steps):
            at = start + datetime.timedelta(minutes=5 * step)
            result = self.operator.run({"Consumption": 100 + step * 0.01}, "", "device", at)
            if result is not None:
                out.append((at, result))
        return out

    def check(self, start):
        # three local days of 5-minute values: the initial phase of two days, then the time windows,
        # which are determined from whole days
        results = self.results(start, 3 * 24 * 12)
        windows = [r for _, r in results if r.get("time_window")]
        self.assertTrue(windows, "no output of a time window")
        for at, result in results:
            self.assertEqual(at.strftime("%Y-%m-%dT%H:%M:%S.000Z"), result["timestamp"])

    def test_the_output_carries_the_utc_time_in_winter(self):
        self.check(datetime.datetime(2026, 10, 26, 23, 0, 30, tzinfo=datetime.timezone.utc))

    def test_the_output_carries_the_utc_time_in_summer(self):
        self.check(datetime.datetime(2026, 6, 30, 22, 0, 30, tzinfo=datetime.timezone.utc))


if __name__ == '__main__':
    unittest.main()
