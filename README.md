# Consumption Profile
## Input 
| key                | type | description | 
|--------------------|------|-------------|   
| `Consumption`     | float | Numeric value that describes any kind of consumption. |
| `Time`     | string | Timestamp of consumption value |



## Output 

| key | type | description | 
|--------------------|-------------|-----------------------------------------------------------| 
| `value`           | int | 1 if the consumption during the last time window was unusual; 0 otherwise. |
| `timestamp`           | string | This string includes the timestamp of the last datapoint. |
| `type`           | string | "low" if unusually low consumption was detected during the last time window, "high" if unusually high consumption was detected, "" or absent otherwise|
| `deviation`           | float | Deviation of the consumption during the last time window from the median of its earlier consumptions, in percent. Absent with fewer than 7 earlier consumptions or when their median is 0. |
| `last_consumptions`           | string | This string represents a pandas dataframe in which the consumptions from the last days during the depicted time window of the day are stored. |
| `time_window`           | string | This string includes the time boundaries from the current time window of the day. |
| `initial_phase`           | string | This string includes an information about whether the operator is in an initial learning phase or not. |


## Config options

| key | type | description | 
|--------------------|-------------|-----------------------------------------------------------| 
| `logger_level`           | str | default: "warning" |
| `init_phase_length`           | int |  |
| `init_phase_level`           | string | |
| `mad_factor`           | float | default: 3. See below. |
| `min_deviation_percent`           | float | default: 25. See below. |

Example: If init_phase_length is 14 and init_phase_level is "d" then the initial phase lasts 14 days.

## When a time window is unusual

The consumption during a time window is compared with up to 14 earlier consumptions of the same time window. It is unusual if it lies more than `mad_factor` median absolute deviations (scaled to the standard deviation) away from their median and deviates from the median by at least `min_deviation_percent` percent. In a time window whose median is 0, any consumption is unusually high. With fewer than 7 earlier consumptions nothing is reported.
