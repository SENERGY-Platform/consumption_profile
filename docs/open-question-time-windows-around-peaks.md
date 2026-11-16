# Time windows around the consumption peaks

## Scope

Applies to the time windows of v1.2.0, determined by `window_determination` in [algo/dynamic_window_determination_user_profile.py](../algo/dynamic_window_determination_user_profile.py). It is about where the window boundaries lie, not about how a window is judged; that is settled and described in the [README](../README.md#when-a-time-window-is-unusual). Re-check this when the window determination changes.

Status on 2026-11-16: measured. Today's windows stay. This replaces the open question that stood here.

## How the boundaries are determined today

For weekdays and weekends separately, from the last 90 days:

1. The consumption of each ten-minute slot is averaged over the days and smoothed (Savitzky–Golay, 20 slots, order 2). Incomplete days and days with a clock change are left out.
2. `scipy.signal.find_peaks` finds the peaks (at least 200 minutes apart, prominence at least a 25th of the highest value). Midnight and the left and right bases of the peaks become boundaries; boundaries less than 30 minutes apart are thinned out.
3. A window longer than 3 hours is split into equal parts.

## What was observed

On synthetic data the bases of broad peaks lay far to the left and right of the peaks. The 3-hour rule then split the stretch between them into windows of nearly equal length, about 2.5 hours, and one peak lay close to a boundary. A boundary through a peak makes the consumption of both windows depend on whether the household is a little earlier or later that day.

That still happens on real households - the dynamic windows come out 2.3 to 2.7 hours long on average, so the 3-hour rule does split - but it does not make the window consumptions less comparable. See the result below.

## The questions, and their answers

1. **Do today's dynamic windows make the window consumptions more comparable than fixed 3-hour windows at all? No.** They win on three of six households and lose on three; their median spread is 0.467 against 0.436 for fixed windows.
2. **Do windows placed around the peaks themselves, for example a peak ± half its width (`scipy.signal.peak_widths`), do better than both? Not measurably.** Median spread 0.429, against 0.436 for fixed and 0.467 for dynamic windows, but three wins to three against the dynamic windows and two to three with one tie against the fixed ones.

The differences between the variants are a few percent of the spread. The differences between households are a factor of four. By the decision rule below, no variant replaces today's windows.

## The measurement

The three variants were run against the same meter readings by a script written for this measurement, which is not kept in the repository. For the dynamic variant it called `window_determination` itself, so that variant was the operator's own code rather than a reimplementation. The peak variant placed a boundary at each edge of a peak at half its prominence (`scipy.signal.peak_widths`, `rel_height=0.5`), with midnight added and the same thinning below 30 minutes and splitting above 3 hours as in v1.2.0. Everything downstream of the boundaries - the ten-minute slots, the days left out, the consumption of a window as the difference between its last and its first reading, and `judge` with `mad_factor` 3 and `min_deviation_percent` 25 - was the operator's.

**Data:** Open Power System Data, *Household Data*, version 2020-04-15, licensed CC-BY 4.0. The six `DE_KN_residential*_grid_import` series are cumulative kWh meter readings of six German households (CoSSMic, Konstanz), at one-minute resolution over 495 to 1283 days. They are real measurements, and the same form - one series per household of `time` and a meter reading - is what an export of this platform's own meters yields, so the measurement can be repeated against one unchanged.

**Procedure:** per household the boundaries are determined once from the first 90 days, and the 365 days after them are judged with those boundaries. The spread is, per time window and type of day, the median absolute deviation of its consumptions relative to their median, aggregated as the median over the windows.

| household | fixed 3 h | dynamic | around the peaks |
|---|---|---|---|
| residential1 | 0.180 | **0.167** | 0.182 |
| residential2 | 0.259 | 0.247 | **0.233** |
| residential3 | 0.442 | 0.492 | **0.426** |
| residential4 | 0.843 | **0.776** | 0.845 |
| residential5 | **0.431** | 0.442 | **0.431** |
| residential6 | **0.514** | 0.623 | 0.631 |
| **median** | **0.436** | 0.467 | **0.429** |
| share reported unusual | 0.114 | 0.127 | 0.124 |
| windows per day | 8.0 | 10.0 | 10.5 |

## What this does not answer

- The households are real but not on this platform: Konstanz, 2015 to 2019. `grid_import` is what the meter at the grid connection reads, so for a household with PV its own consumption of its own generation is not in it.
- The variants do not have the same number of windows. Shorter windows hold less consumption, which tends to raise the relative spread, so the comparison is slightly tilted in favour of the fixed windows. Neither of the other two is measurably better even so.
- The boundaries were determined once rather than re-determined at the end of every month as the operator does. Re-determining would let the three variants drift apart differently, and the comparison would no longer be about where the boundaries lie.
- Real data has no labels saying which day was really unusual. Fewer or more reports prove nothing on their own; the spread is the measure that needs no labels.

## Decision

A new variant replaces today's windows only if it is measurably better. None is, so today's windows stay.

Fixed 3-hour windows are not measurably worse than today's either, and dropping the window determination would be less code and less state. That is not what was decided: the dynamic determination is kept because following the household's own day is the better model of what a time window is meant to be, and the measurement gives no reason to go either way.

Repeat the measurement against an export of this platform's meters once one holds more than 90 days, and against any change to the window determination.
