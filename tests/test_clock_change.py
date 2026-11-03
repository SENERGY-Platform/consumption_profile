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
import json
import tempfile
import unittest

from main import CustomConfig, Operator

UTC = datetime.timezone.utc
NIGHT = "00:00:00-03:00:00"


class TestClockChange(unittest.TestCase):
    # On October 25, 2026, a Sunday, the clocks go back from 03:00 to 02:00, so the night window lasts four hours.

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.operator = Operator()
        self.operator.produce = lambda value: None
        self.operator.data_path = self.dir.name
        self.operator.init(None, None, None, None, None, None,
                           config=CustomConfig({"init_phase_length": "2",
                                                "init_phase_level": "d"}))

    def tearDown(self):
        self.dir.cleanup()

    def feed(self, start, end):
        # an equal meter reading every 5 minutes of real time; returns the outputs of the night windows
        # by the day they completed on
        outputs = {}
        at, meter = start, 100.0
        while at < end:
            meter += 0.25
            result = self.operator.run({"Consumption": meter}, "", "device", at)
            if result is not None and result.get("time_window") == NIGHT:
                outputs[at.day] = result
            at += datetime.timedelta(minutes=5)
        return outputs

    def test_does_not_judge_the_night_of_a_clock_change(self):
        # from Saturday, September 26, 00:00:30 local time; nine weekend nights before the clock change
        nights = self.feed(datetime.datetime(2026, 9, 25, 22, 0, 30, tzinfo=UTC),
                           datetime.datetime(2026, 10, 25, 6, 0, tzinfo=UTC))
        self.assertEqual((0, None), (nights[25]["value"], nights[25].get("deviation")))

    def test_leaves_out_the_night_of_a_clock_change_that_falls_on_a_month_end(self):
        # October 31, 2027 is a Sunday, the clocks go back that night, and the windows are determined anew on its
        # first value; the next weekend night is on Saturday, November 6
        nights = self.feed(datetime.datetime(2027, 10, 26, 22, 0, 30, tzinfo=UTC),
                           datetime.datetime(2027, 11, 6, 4, 0, tzinfo=UTC))
        consumptions = [consumption for _, consumption, _ in json.loads(nights[6]["last_consumptions"])]
        # Saturday, October 30, and November 6; nothing for the four-hour night in between
        self.assertEqual([9000.0, 9000.0], consumptions)


if __name__ == '__main__':
    unittest.main()
