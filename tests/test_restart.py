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
# A power of two, so that every window consumption is exact and the normal ones are equal.
NORMAL = 0.25


def local(day, hour, minute=0):
    # June 2026 has no month end and no clock change in the days used here; the messages
    # arrive 30 s past every fifth minute.
    return datetime.datetime(2026, 6, day, hour, minute, 30, tzinfo=CEST)


class TestRestart(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.timescale = Timescale()
        self.at = local(1, 0)
        self.meter = 100.0
        self.operator = self.start()

    def tearDown(self):
        self.dir.cleanup()

    def start(self):
        operator = Operator()
        operator.produce = lambda value: None
        operator.data_path = self.dir.name
        with self.timescale.patched():
            operator.init(None, None, None, None, None, None,
                          config=CustomConfig({"init_phase_length": "2",
                                               "init_phase_level": "d"}))
        return operator

    def restart(self, new_volume=False):
        # a new operator, as after a crash or an update; it reads the history from Timescale
        if new_volume:
            self.dir.cleanup()
            self.dir = tempfile.TemporaryDirectory()
        self.operator = self.start()

    def feed(self, until, rate=NORMAL, down=False):
        # a meter value every 5 minutes, which Timescale always gets and the operator unless it is down;
        # returns the outputs of the windows completed meanwhile
        outputs = []
        while self.at < until:
            self.meter += rate
            self.timescale.write(self.at, self.meter)
            if not down:
                result = self.operator.run({"Consumption": self.meter}, "", "device", self.at)
                if result is not None and result.get("time_window"):
                    outputs.append(result)
            self.at += datetime.timedelta(minutes=5)
        return outputs

    def window(self, outputs, time_window):
        return [(r["value"], r.get("type", ""), r.get("deviation")) for r in outputs if r["time_window"] == time_window]

    def test_reports_an_unusually_high_consumption_after_a_restart(self):
        self.feed(until=local(11, 13, 30))
        self.restart()
        self.feed(until=local(12, 12))
        outputs = self.feed(until=local(12, 15, 5), rate=2 * NORMAL)
        self.assertEqual([(1, "high", 100.0)], self.window(outputs, "12:00:00-15:00:00"))

    def test_judges_the_window_a_restart_cuts_through_by_its_whole_consumption(self):
        self.feed(until=local(11, 13, 30))
        self.restart()
        outputs = self.feed(until=local(11, 15, 5))
        self.assertEqual([(0, "", 0.0)], self.window(outputs, "12:00:00-15:00:00"))

    def test_judges_at_once_on_a_new_volume(self):
        self.feed(until=local(11, 13, 30))
        self.restart(new_volume=True)
        outputs = self.feed(until=local(11, 15, 5))
        self.assertEqual([(0, "", 0.0)], self.window(outputs, "12:00:00-15:00:00"))

    def test_compares_with_the_windows_that_passed_while_it_was_down(self):
        self.feed(until=local(11, 13, 30))
        self.feed(until=local(11, 19), down=True)
        self.restart()
        self.feed(until=local(11, 19, 10))
        entries = self.operator.time_window_consumption_list_dict[("weekday", "15:00:00-18:00:00")]
        self.assertEqual((datetime.date(2026, 6, 11), 9000.0), (entries[-1][0].date(), entries[-1][1]))


if __name__ == '__main__':
    unittest.main()
