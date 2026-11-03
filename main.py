"""
   Copyright 2022 InfAI (CC SES)

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

__all__ = ("Operator", )

from operator_lib.util import OperatorBase, logger, InitPhase, todatetime, timestamp_to_str
from operator_lib.util.persistence import save, load
from operator_lib.util.config import MissingConfigValueError
from operator_lib.util.helpers import provide_historic_frames_local, HistoryNeedsRayError
import pandas as pd
import numpy as np
import os
import datetime
from collections import defaultdict
from algo import dynamic_window_determination_user_profile as dwdup
from algo.judgement import judge
import typing
import datetime

from operator_lib.util import Config, Inputs, Outputs
class CustomConfig(Config):
    init_phase_length: float = 2
    init_phase_level: str = "d"
    mad_factor: float = 3
    min_deviation_percent: float = 25
    # A field of the library, declared again so that flows can set it.
    logger_level: str = "warning"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 2
        
        if self.init_phase_level == '':
            self.init_phase_level = 'd'

        if self.mad_factor != '':
            self.mad_factor = float(self.mad_factor)
        else:
            self.mad_factor = 3

        if self.min_deviation_percent != '':
            self.min_deviation_percent = float(self.min_deviation_percent)
        else:
            self.min_deviation_percent = 25

# The time of a value is the time the platform received it, so there is no input for it.
class ConsumptionInputs(Inputs):
    Consumption: float

class ProfileOutputs(Outputs):
    value: int
    timestamp: str
    type: str
    deviation: float
    last_consumptions: str
    time_window: str
    initial_phase: str

class Operator(OperatorBase):
    configType = CustomConfig
    inputs = ConsumptionInputs
    outputs = ProfileOutputs
    # Where the state is kept, the volume of the deployment; not a config value, which a flow could change.
    data_path = "/opt/data"
    # This much history is read at the start, and the time windows are determined from it; older values are dropped then.
    history_length = pd.Timedelta(90, 'd')

    def init(self,  *args, **kwargs):
        super().init(*args, **kwargs)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        self.data_history = self.read_history()
        self.first_data_time = load(self.data_path, "first_data_time.pickle")
        if not self.data_history.empty and (not self.first_data_time or self.data_history.index[0] < self.first_data_time):
            # The initial phase counts from the first value of the history, so a history of its length ends it at once.
            self.first_data_time = self.data_history.index[0]
            save(self.data_path, "first_data_time.pickle", self.first_data_time)

        self.time_window_data_just_updated = False



        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)

        # No message before the first value: without a time of its own the export drops it, and the end of the
        # initial phase is not known yet. The first value announces the initial phase.
        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)

        self.consumption_same_time_window = []

        # Both per type of day, and like everything derived from the history not kept: the first message after the
        # initial phase rebuilds them from it, so they also hold the windows that passed while the operator was down.
        self.window_boundaries = {day_type: [datetime.time(0, 0), datetime.time(3, 0), datetime.time(6, 0),  datetime.time(9, 0), datetime.time(12, 0), datetime.time(15, 0),
                                  datetime.time(18, 0), datetime.time(21, 0)] for day_type in dwdup.DAY_TYPES}
        # (time, consumption, judgement) per (type of day, window).
        self.time_window_consumption_list_dict = defaultdict(list)

    def read_history(self):
        # Timescale is the only source of the history: it is read at every start and extended live, so a restart or a
        # new volume loses nothing. Any other failure of the read fails the start, and the platform starts it again.
        no_history = pd.Series([], index=[], dtype=object)
        try:
            frames = provide_historic_frames_local(self.history_length)
        except (HistoryNeedsRayError, MissingConfigValueError) as e:
            logger.warning(f'Starting without history: {e}')
            return no_history
        if len(frames) != 1 or 'device_id' in frames[0].columns:
            logger.warning('Starting without history: the consumption comes from more than one device')
            return no_history
        frame = frames[0].dropna(subset=['Consumption'])
        # As live: German time without a timezone, and a value no later than every value before it is left out.
        index = pd.DatetimeIndex(pd.to_datetime(frame['time'], utc=True)).tz_convert(dwdup.TIMEZONE).tz_localize(None)
        history = pd.Series(frame['Consumption'].astype(float).to_numpy(), index=index)
        latest_before = history.index.to_series().cummax().shift().fillna(pd.Timestamp.min)
        return history[history.index > latest_before.to_numpy()]

    def create_new_time_window_consumption_list_dict(self):
        self.time_window_consumption_list_dict = defaultdict(list)
        # From the whole history, since 14 weekend days span seven weeks.
        for day_type in dwdup.DAY_TYPES:
            for day, windows in dwdup.window_division(self.window_boundaries[day_type], self.data_history):
                if dwdup.type_of_day(day) != day_type:
                    continue
                for window_name, window in windows:
                    # A window during which the clocks change lasts an hour more or less and is left out, as live.
                    if list(window)!=[] and not dwdup.spans_clock_change(window.index[0], window.index[-1]):
                        overall_time_window_consumption = 1000*(window.iloc[-1]-window.iloc[0])
                        if overall_time_window_consumption>=0:
                            self.append_judged((day_type, window_name), window.index[-1], overall_time_window_consumption)

    def append_judged(self, window, time, consumption):
        entries = self.time_window_consumption_list_dict[window]
        judgement, deviation = judge([earlier for _, earlier, _ in entries[-14:]], consumption,
                                     self.config.mad_factor, self.config.min_deviation_percent)
        entries.append((time, consumption, judgement))
        return judgement, deviation

    def update_time_window_data(self):
        self.data_history = self.data_history[self.timestamp-self.history_length:]
        for day_type in dwdup.DAY_TYPES:
            self.window_boundaries[day_type] = dwdup.window_determination(self.data_history, self.window_boundaries[day_type], day_type)
        self.create_new_time_window_consumption_list_dict()

    def update_time_window_consumption_list_dict(self):
        if (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > 3*pd.Timedelta(1,'h')) or (
           (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) < 0*pd.Timedelta(1,'h')) and (
            pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > -21*pd.Timedelta(1,'h')
           ) 
        ):
            return
        if dwdup.spans_clock_change(self.consumption_same_time_window[0]['Time'], self.consumption_same_time_window[-1]['Time']):
            # The clocks changed during this window, so it lasted an hour more or less than on other days. It is left out,
            # together with the part of it that an update right before recorded from the history.
            if self.time_window_data_just_updated == True:
                entries = self.time_window_consumption_list_dict[self.completed_window]
                if entries and entries[-1][0] == self.consumption_same_time_window[0]['Time']:
                    entries.pop()
                self.time_window_data_just_updated = False
            return
        time_window_consumption_max = float(self.consumption_same_time_window[-1]['Consumption'])
        time_window_consumption_min = float(self.consumption_same_time_window[0]['Consumption'])
        overall_time_window_consumption = 1000*(time_window_consumption_max-time_window_consumption_min)
        judged = None
        if np.isnan(overall_time_window_consumption)==False and overall_time_window_consumption >= 0:
            if self.time_window_data_just_updated == False:
                judged = self.append_judged(self.completed_window, self.timestamp, overall_time_window_consumption)
            elif self.time_window_data_just_updated == True:
                # If in the current time window the time_window_data was updated there is already an entry for the current time window from the current day. The consumption
                # that is stored there depicts the consumption from the window until the point of time_window_data updating. The rest of the consumption during the window has just
                # to be added.
                _, consumption_before_update, _ = self.time_window_consumption_list_dict[self.completed_window].pop()
                judged = self.append_judged(self.completed_window, self.timestamp, consumption_before_update+overall_time_window_consumption)
                self.time_window_data_just_updated = False
        return judged

    def run(self, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        # Convert to german time and then forget the timezone. The time windows are of the local
        # day; the output carries the UTC time, since the export reads it as UTC.
        utc = pd.Timestamp(timestamp).tz_convert("UTC")
        self.timestamp = utc.tz_convert(dwdup.TIMEZONE).tz_localize(None)
        data = {'Consumption': data['Consumption'], 'Time': self.timestamp}
        if not self.first_data_time:
            self.first_data_time = self.timestamp
            self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
            save(self.data_path, "first_data_time.pickle", self.first_data_time)
        logger.debug('energy: '+str(data['Consumption'])+'  '+'time: '+str(self.timestamp))

        if len(self.data_history.index) >= 1 and self.timestamp <= self.data_history.index[-1]: # Discard points that come in wrong order wrt time.
            return
        self.data_history = pd.concat([self.data_history, pd.Series([float(data['Consumption'])], index=[self.timestamp])])

        operator_is_init = self.init_phase_handler.operator_is_in_init_phase(self.timestamp)
        
        
        
        init_value = {
                    "value": 0,
                    "timestamp": timestamp_to_str(utc),
                    "type": "",
                    "last_consumptions": "",
                    "time_window": ""
        }
        if operator_is_init:
            return self.init_phase_handler.generate_init_msg(self.timestamp, init_value)

        if self.init_phase_handler.init_phase_needs_to_be_reset():
            return self.init_phase_handler.reset_init_phase(init_value)

        if not self.time_window_consumption_list_dict or (self.timestamp.is_month_end and self.timestamp.date() > self.data_history.index[-2].date()):# Update time window data right after inital phase and then after each month.
            self.update_time_window_data()
            self.time_window_data_just_updated = True
            logger.debug(self.window_boundaries)
            self.consumption_same_time_window = []
        self.current_time_window_start = max(time for time in self.window_boundaries[dwdup.type_of_day(self.timestamp)] if time<=self.timestamp.time())
        if self.consumption_same_time_window == []:
            self.consumption_same_time_window.append(data)
        elif self.consumption_same_time_window != []:
            previous_time = self.consumption_same_time_window[-1]['Time']
            self.last_time_window_start = max(time for time in self.window_boundaries[dwdup.type_of_day(previous_time)] if time<=previous_time.time())
            if self.current_time_window_start==self.last_time_window_start:
                self.consumption_same_time_window.append(data)
            elif (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > 3*pd.Timedelta(1,'h')) or (  # If the gap between the current and last window boundary is too large 
                 (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) < 0*pd.Timedelta(1,'h')) and ( # (because for example there were data points missing in the stream) we ignore the current consumption
                  pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > -21*pd.Timedelta(1,'h'))):   # and start again.
                self.consumption_same_time_window = [data] 
                return
            else:
                self.consumption_same_time_window.append(data) #!!! Otherwise I lose energy which is consumed between two consecutive windows.
                # A window that runs past midnight belongs to the type of the day it started on.
                self.completed_window = (dwdup.type_of_day(self.consumption_same_time_window[0]['Time']),
                                         f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}')
                judgement, deviation = self.update_time_window_consumption_list_dict() or ("", None)
                self.consumption_same_time_window = [data]
                if judgement == "low":
                    logger.warning(f'In letzter Zeit wurde ungewöhnlich wenig verbraucht.')
                elif judgement == "high":
                    logger.warning(f'In letzter Zeit wurde ungewöhnlich viel verbraucht.')
                return self.create_output(judgement, deviation, utc, self.create_df_cons_last_14_days())

    def create_df_cons_last_14_days(self):
        entries = self.time_window_consumption_list_dict[self.completed_window][-14:]
        df = pd.DataFrame({0:[consumption for _, consumption, _ in entries], 1:[1 if judgement else 0 for _, _, judgement in entries]},
                          index=[timestamp.date() for timestamp, _, _ in entries])
        return df.reset_index(inplace=False).to_json(orient="values")

    def create_output(self, judgement, deviation, timestamp, df_cons_last_14_days):
        output = {
                    "value": 1 if judgement else 0,
                    "timestamp": timestamp_to_str(timestamp),
                    "last_consumptions": df_cons_last_14_days,
                    "time_window": f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'
        }
        if judgement:
            output["type"] = judgement
        if deviation is not None:
            output["deviation"] = round(deviation, 1)
        return output

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="user-profile-operator", git_info_file='git_commit')
