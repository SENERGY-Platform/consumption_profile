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


class TestInitialPhase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.sent = []
        self.operator = Operator()
        self.operator.produce = self.sent.append
        self.operator.init(None, None, None, None, None, None,
                           config=CustomConfig({"data_path": self.dir.name, "init_phase_length": "2",
                                                "init_phase_level": "d"}))

    def tearDown(self):
        self.dir.cleanup()

    def test_sends_nothing_before_the_first_value(self):
        # a message without a time of its own never reaches the export
        self.assertEqual([], self.sent)

    def test_announces_the_end_of_the_initial_phase_with_the_first_value(self):
        result = self.operator.run({"Consumption": 100}, "", "device", datetime.datetime(2026, 6, 1, 10, 0, 30, tzinfo=CEST))
        self.assertEqual("2026-06-01T08:00:30.000Z", result["timestamp"])
        self.assertIn("2026-06-03T10:00:30", result["initial_phase"])


if __name__ == '__main__':
    unittest.main()
