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

__all__ = ("judge", "MIN_EARLIER_VALUES")

import numpy as np

# With fewer earlier values of a window there is no usual consumption to compare with.
MIN_EARLIER_VALUES = 7
# Scales the median absolute deviation to the standard deviation of normally distributed values.
MAD_TO_STD = 1.4826


def judge(earlier, value, mad_factor, min_deviation_percent):
    """
    Judges the consumption of a window against the earlier consumptions of the same window: "high" or
    "low" when it lies more than mad_factor scaled median absolute deviations and at least
    min_deviation_percent away from their median, "" otherwise. Returns that together with the deviation
    from the median in percent, which is None without enough earlier values or with a median of 0.
    """
    if len(earlier) < MIN_EARLIER_VALUES:
        return "", None
    median = float(np.median(earlier))
    spread = mad_factor * MAD_TO_STD * float(np.median(np.abs(np.asarray(earlier, dtype=float) - median)))
    if median > 0:
        deviation = (value - median) / median * 100
        large_enough = abs(deviation) >= min_deviation_percent
    else:
        # Usually nothing is consumed in this window, so any consumption is a deviation.
        deviation = None
        large_enough = value > 0
    if abs(value - median) <= spread or not large_enough:
        return "", deviation
    return ("high" if value > median else "low"), deviation
