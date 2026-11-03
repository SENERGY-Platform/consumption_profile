# Consumption Profile
## Input 
| key                | type | description | 
|--------------------|------|-------------|   
| `Consumption`     | float | Numeric value that describes any kind of consumption. |

The time of a value is the time the platform received it.



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
| `init_phase_length`           | float | default: 2 |
| `init_phase_level`           | string | default: "d" |
| `mad_factor`           | float | default: 3. See below. |
| `min_deviation_percent`           | float | default: 25. See below. |

Example: If init_phase_length is 14 and init_phase_level is "d" then the initial phase lasts 14 days.

The operator keeps the state of its initial phase in `/opt/data`, which should be a volume. `operator-manifest.json` declares the inputs, outputs and config options; the CI regenerates it from `main.py` before each release.

## History

Timescale is the only source of the history. At every start the operator reads the last 90 days of its input from there (through `ts_conn`, which the flow engine gives a deployed operator) and extends them with the values it receives. Its time windows and the earlier consumptions it compares with are derived from that anew at every start, so a restart, an update or a new volume loses nothing, and a new deployment judges from its first value on if Timescale already holds enough of it. The initial phase counts from the first value in Timescale.

Values arriving after a restart that are no later than the last value read from Timescale are left out: the time windows that passed while the operator was down get no output of their own, but count among the earlier consumptions.

An input whose history Timescale does not have, such as the output of another operator, or a deployment without a connection to Timescale, starts without history and learns from the values it receives; it logs a warning and starts from scratch after every restart. If Timescale cannot be read at all, the start fails and the platform starts the operator again.

## When a time window is unusual

Weekdays (Monday to Friday) and weekends have time windows of their own, determined at every start and each month from the respective days of the last 90 days. A time window that runs past midnight belongs to the day it started on.

The consumption during a time window is compared with up to 14 earlier consumptions of the same time window on the same type of day. It is unusual if it lies more than `mad_factor` median absolute deviations (scaled to the standard deviation) away from their median and deviates from the median by at least `min_deviation_percent` percent. In a time window whose median is 0, any consumption is unusually high. With fewer than 7 earlier consumptions nothing is reported, so without enough history in Timescale weekends are only judged about four weeks after the initial phase. A time window during which the clocks change lasts an hour more or less and is not judged; days with a clock change are left out when the time windows are determined.
