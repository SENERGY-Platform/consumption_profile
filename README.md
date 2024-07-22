# Consumption Profile
## Input 
| key                | type | description | 
|--------------------|------|-------------|   
| `Consumption`     | float | Numeric value that describes any kind of consumption. |
| `Time`     | string | Timestamp of consumption value |



## Output 

| key | type | description | 
|--------------------|-------------|-----------------------------------------------------------| 
| `value`           | int | 0 if anomalous consumption was detected during the last time window; 1 otherwise. |
| `timestamp`           | string | This string includes the timestamp of the last datapoint. |
| `type`           | string | "low" if unusually low consumption was detected during the last time window, "high" if unusually high consumption was detected, "" otherwise|
| `last_consumptions`           | string | This string represents a pandas dataframe in which the consumptions from the last days during the depicted time window of the day are stored. |
| `time_window`           | string | This string includes the time boundaries from the current time window of the day. |
| `initial_phase`           | string | This string includes an information about whether the operator is in an initial learning phase or not. |


## Config options

| key | type | description | 
|--------------------|-------------|-----------------------------------------------------------| 
| `logger_level`           | str | default: "warning" |
| `init_phase_length`           | int |  |
| `init_phase_level`           | string | |

Example: If init_phase_length is 14 and init_phase_level is "d" then the initial phase lasts 14 days.
