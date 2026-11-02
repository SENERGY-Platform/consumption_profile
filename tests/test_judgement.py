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

import unittest

from algo.judgement import judge

USUAL = [90, 95, 100, 100, 105, 110, 100, 98]


def judged(earlier, value):
    return judge(earlier, value, mad_factor=3, min_deviation_percent=25)


class TestJudgement(unittest.TestCase):
    def test_does_not_judge_a_window_with_fewer_than_seven_earlier_values(self):
        self.assertEqual(("", None), judged([100] * 6, 500))

    def test_reports_a_consumption_far_above_the_usual_as_high(self):
        self.assertEqual(("high", 100.0), judged(USUAL, 200))

    def test_reports_a_consumption_far_below_the_usual_as_low(self):
        self.assertEqual(("low", -60.0), judged(USUAL, 40))

    def test_ignores_a_deviation_smaller_than_the_minimum_percentage(self):
        # beyond the spread, which is 0 here, but only 10 %
        self.assertEqual(("", 10.0), judged([100] * 7, 110))

    def test_ignores_a_deviation_within_the_usual_spread(self):
        self.assertEqual(("", 60.0), judged([50, 150, 60, 140, 70, 130, 100], 160))

    def test_reports_any_consumption_in_a_window_that_is_usually_empty(self):
        self.assertEqual(("high", None), judged([0] * 7, 5))


if __name__ == '__main__':
    unittest.main()
