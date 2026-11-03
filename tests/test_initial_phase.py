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

from fake_timescale import Timescale
from main import CustomConfig, Operator

CEST = datetime.timezone(datetime.timedelta(hours=2))


class TestInitialPhase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.sent = []
        self.timescale = Timescale()

    def tearDown(self):
        self.dir.cleanup()

    def start(self):
        operator = Operator()
        operator.produce = self.sent.append
        operator.data_path = self.dir.name
        with self.timescale.patched():
            operator.init(None, None, None, None, None, None,
                          config=CustomConfig({"init_phase_length": "2",
                                               "init_phase_level": "d"}))
        return operator

    def history(self, start, end):
        # meter readings every 5 minutes in Timescale
        at = start
        while at < end:
            self.timescale.write(at, 100.0)
            at += datetime.timedelta(minutes=5)

    def test_sends_nothing_before_the_first_value(self):
        self.start()
        # a message without a time of its own never reaches the export
        self.assertEqual([], self.sent)

    def test_announces_the_end_of_the_initial_phase_with_the_first_value(self):
        result = self.start().run({"Consumption": 100}, "", "device", datetime.datetime(2026, 6, 1, 10, 0, 30, tzinfo=CEST))
        self.assertEqual("2026-06-01T08:00:30.000Z", result["timestamp"])
        self.assertIn("2026-06-03T10:00:30", result["initial_phase"])

    def test_counts_the_initial_phase_from_the_first_value_in_timescale(self):
        self.history(datetime.datetime(2026, 6, 1, 10, 0, 30, tzinfo=CEST), datetime.datetime(2026, 6, 2, 10, 0, 30, tzinfo=CEST))
        result = self.start().run({"Consumption": 100}, "", "device", datetime.datetime(2026, 6, 2, 10, 0, 30, tzinfo=CEST))
        self.assertIn("2026-06-03T10:00:30", result["initial_phase"])

    def test_a_history_as_long_as_the_initial_phase_ends_it_with_the_first_value(self):
        self.history(datetime.datetime(2026, 5, 30, 10, 0, 30, tzinfo=CEST), datetime.datetime(2026, 6, 1, 10, 0, 30, tzinfo=CEST))
        result = self.start().run({"Consumption": 100}, "", "device", datetime.datetime(2026, 6, 1, 10, 0, 30, tzinfo=CEST))
        self.assertEqual("", result["initial_phase"])


if __name__ == '__main__':
    unittest.main()
