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

CEST = datetime.timezone(datetime.timedelta(hours=2))
# A power of two, so that every window consumption is exact and the normal ones are equal.
NORMAL = 0.25


def local(day, hour, minute=0):
    # June 2026 has no month end and no clock change in the days used here; the messages
    # arrive 30 s past every fifth minute.
    return datetime.datetime(2026, 6, day, hour, minute, 30, tzinfo=CEST)


class TestRestart(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.at = local(1, 0)
        self.meter = 100.0
        self.operator = self.start()

    def tearDown(self):
        self.dir.cleanup()

    def start(self):
        operator = Operator()
        operator.produce = lambda value: None
        operator.data_path = self.dir.name
        operator.init(None, None, None, None, None, None,
                      config=CustomConfig({"init_phase_length": "2",
                                           "init_phase_level": "d"}))
        return operator

    def restart(self):
        # as on SIGTERM: stop saves the history, a new operator loads the state
        self.operator.stop()
        self.operator = self.start()

    def feed(self, until, rate=NORMAL):
        # a meter value every 5 minutes; returns the outputs of the windows completed meanwhile
        outputs = []
        while self.at < until:
            self.meter += rate
            result = self.operator.run({"Consumption": self.meter}, "", "device", self.at)
            if result is not None and result.get("time_window"):
                outputs.append(result)
            self.at += datetime.timedelta(minutes=5)
        return outputs

    def window(self, outputs, time_window):
        return [(r["value"], r.get("type", "")) for r in outputs if r["time_window"] == time_window]

    def test_reports_an_unusually_high_consumption_after_a_restart(self):
        self.feed(until=local(11, 13, 30))
        self.restart()
        self.feed(until=local(12, 12))
        outputs = self.feed(until=local(12, 15, 5), rate=2 * NORMAL)
        self.assertEqual([(1, "high")], self.window(outputs, "12:00:00-15:00:00"))

    def test_does_not_report_the_window_a_restart_cuts_short_as_unusually_low(self):
        self.feed(until=local(11, 13, 30))
        self.restart()
        outputs = self.feed(until=local(11, 15, 5))
        self.assertEqual([(0, "")], self.window(outputs, "12:00:00-15:00:00"))


if __name__ == '__main__':
    unittest.main()
