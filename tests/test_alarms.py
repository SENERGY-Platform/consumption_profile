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
# A power of two, so that the window consumptions of equal days are exact and equal.
NORMAL = 0.25


class TestAlarms(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.operator = Operator()
        self.operator.produce = lambda value: None
        self.operator.data_path = self.dir.name
        with Timescale().patched():
            self.operator.init(None, None, None, None, None, None,
                               config=CustomConfig({"init_phase_length": "2",
                                                    "init_phase_level": "d"}))

    def tearDown(self):
        self.dir.cleanup()

    def days(self, factors):
        # one day of 5-minute meter readings for each factor of the normal consumption, from June 1, 2026;
        # returns the outputs of the completed windows by the day they completed on
        outputs = {}
        at = datetime.datetime(2026, 6, 1, 0, 0, 30, tzinfo=CEST)
        meter = 100.0
        for factor in factors:
            for _ in range(288):
                meter += NORMAL * factor
                result = self.operator.run({"Consumption": meter}, "", "device", at)
                if result is not None and result.get("time_window"):
                    outputs.setdefault(at.day, []).append(result)
                at += datetime.timedelta(minutes=5)
        return outputs

    def reported(self, outputs):
        return {day: [r["time_window"] for r in results if r["value"] == 1] for day, results in outputs.items()
                if any(r["value"] == 1 for r in results)}

    def test_reports_nothing_before_a_window_has_seven_earlier_values(self):
        # two days of initial phase, then days that vary as ordinary days do
        outputs = self.days([1.0, 1.0, 0.93, 1.09, 0.96, 0.99, 1.07])
        self.assertEqual({}, self.reported(outputs))

    def test_reports_no_day_that_is_only_nine_percent_below_the_usual(self):
        outputs = self.days([1.0, 1.0, 0.93, 1.09, 0.96, 0.99, 1.07, 0.98, 1.01, 0.91])
        self.assertEqual({}, self.reported(outputs))

    def weeks(self, last_day_factor):
        # June 1 to 28, 2026: weekends consume twice as much as weekdays; June 28 is the eighth weekend day
        factors = [2.0 if datetime.date(2026, 6, day).weekday() >= 5 else 1.0 for day in range(1, 29)]
        return self.days(factors[:-1] + [last_day_factor])

    def test_does_not_report_weekends_that_consume_as_much_as_earlier_weekends(self):
        self.assertEqual({}, self.reported(self.weeks(last_day_factor=2.0)))

    def test_reports_a_weekend_day_that_consumes_half_as_much_as_earlier_weekends(self):
        midday = [r for r in self.weeks(last_day_factor=1.0)[28] if r["time_window"] == "12:00:00-15:00:00"]
        self.assertEqual([(1, "low", -50.0)], [(r["value"], r.get("type"), r.get("deviation")) for r in midday])

    def test_reports_a_day_with_twice_the_usual_consumption_with_its_deviation(self):
        outputs = self.days([1.0] * 9 + [2.0])
        midday = [r for r in outputs[10] if r["time_window"] == "12:00:00-15:00:00"]
        self.assertEqual([(1, "high", 100.0)], [(r["value"], r.get("type"), r.get("deviation")) for r in midday])


if __name__ == '__main__':
    unittest.main()
