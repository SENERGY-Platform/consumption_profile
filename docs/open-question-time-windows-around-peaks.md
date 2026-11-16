# Open question: time windows around the consumption peaks

## Scope

Applies to the time windows of v1.2.0, determined by `window_determination` in [algo/dynamic_window_determination_user_profile.py](../algo/dynamic_window_determination_user_profile.py). It is about where the window boundaries lie, not about how a window is judged; that is settled and described in the [README](../README.md#when-a-time-window-is-unusual). Re-check this when the window determination changes.

Status on 2026-11-16: open, waiting for an export of real meter readings.

## How the boundaries are determined today

For weekdays and weekends separately, from the last 90 days:

1. The consumption of each ten-minute slot is averaged over the days and smoothed (Savitzky–Golay, 20 slots, order 2). Incomplete days and days with a clock change are left out.
2. `scipy.signal.find_peaks` finds the peaks (at least 200 minutes apart, prominence at least a 25th of the highest value). Midnight and the left and right bases of the peaks become boundaries; boundaries less than 30 minutes apart are thinned out.
3. A window longer than 3 hours is split into equal parts.

## What was observed

On synthetic data the bases of broad peaks lay far to the left and right of the peaks. The 3-hour rule then split the stretch between them into windows of nearly equal length, about 2.5 hours, and one peak lay close to a boundary. A boundary through a peak makes the consumption of both windows depend on whether the household is a little earlier or later that day. Whether real households show the same has not been checked.

## The question

1. Do today's dynamic windows make the window consumptions more comparable than fixed 3-hour windows at all?
2. Do windows placed around the peaks themselves, for example a peak ± half its width (`scipy.signal.peak_widths`), do better than both?

## How to test it

- **Data:** one CSV per household with `time` (UTC, ISO 8601) and `value` (meter reading), at least 3 months, ideally 5–10 households.
- **Variants:** fixed 3-hour windows, today's dynamic windows and windows around the peaks, each evaluated on the same data. The judgement stays as in v1.2.0 with its defaults, so only the windows differ.
- **Measures per variant:**
  - the spread of the consumptions of each window and type of day, relative to their median: less spread means more comparable values;
  - the share of windows reported as unusual.
- **Caveat:** Real data has no labels saying which day was really unusual. Fewer reports alone do not prove fewer false alarms; the spread is the measure that needs no labels.
- **Decision:** A new variant replaces today's windows only if it is measurably better. The result then replaces this question here, also if today's windows stay.
