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
import pandas as pd
import numpy as np
import os
import datetime
from collections import defaultdict
from algo import dynamic_window_determination_user_profile as dwdup
from algo.judgement import judge
import typing
import datetime

from operator_lib.util import Config
class CustomConfig(Config):
    data_path = "/opt/data"
    init_phase_length: float = 2
    init_phase_level: str = "d"
    mad_factor: float = 3
    min_deviation_percent: float = 25

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

class Operator(OperatorBase):
    configType = CustomConfig
    # The time windows are determined from this much history; older values are dropped then.
    history_length = pd.Timedelta(90, 'd')

    def init(self,  *args, **kwargs):
        super().init(*args, **kwargs)
        self.data_path = self.config.data_path
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        self.first_data_time = load(self.config.data_path, "first_data_time.pickle")

        self.time_window_data_just_updated = None



        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)

        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        value = {
                    "value": 0,
                    "timestamp": "",
                    "type": "",
                    "last_consumptions": "",
                    "time_window": ""
        }
        self.init_phase_handler.send_first_init_msg(value)

        self.consumption_same_time_window = []

        # Both per type of day. Files of their own, since the earlier files hold other entries; without them the windows
        # are rebuilt from the history on the first message.
        self.window_boundaries = load(self.data_path, 'window_boundaries_by_day_type.pickle', default={day_type: [datetime.time(0, 0), datetime.time(3, 0), datetime.time(6, 0),  datetime.time(9, 0), datetime.time(12, 0), datetime.time(15, 0),
                                         datetime.time(18, 0), datetime.time(21, 0)] for day_type in dwdup.DAY_TYPES})
        self.data_history = load(self.data_path, "data_history.pickle", default=pd.Series([], index=[],dtype=object))
        # (time, consumption, judgement) per (type of day, window).
        self.time_window_consumption_list_dict = load(self.data_path, "time_window_consumptions_by_day_type.pickle", default=defaultdict(list))

    def create_new_time_window_consumption_list_dict(self):
        self.time_window_consumption_list_dict = defaultdict(list)
        # From the whole history, since 14 weekend days span seven weeks.
        for day_type in dwdup.DAY_TYPES:
            for day, windows in dwdup.window_division(self.window_boundaries[day_type], self.data_history):
                if dwdup.type_of_day(day) != day_type:
                    continue
                for window_name, window in windows:
                    if list(window)!=[]:
                        overall_time_window_consumption = 1000*(window.iloc[-1]-window.iloc[0])
                        if overall_time_window_consumption>=0:
                            self.append_judged((day_type, window_name), window.index[-1], overall_time_window_consumption)
        save(self.data_path, "time_window_consumptions_by_day_type.pickle", self.time_window_consumption_list_dict)

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
        save(self.data_path, 'window_boundaries_by_day_type.pickle', self.window_boundaries)
        self.create_new_time_window_consumption_list_dict()

    def update_time_window_consumption_list_dict(self):
        if (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > 3*pd.Timedelta(1,'h')) or (
           (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) < 0*pd.Timedelta(1,'h')) and (
            pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > -21*pd.Timedelta(1,'h')
           ) 
        ):
            return
        if self.time_window_data_just_updated is None:
            # init leaves the flag at None: after a restart this window lacks its values from before the
            # restart, so its consumption would be too low. Only the windows after it are recorded.
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
        save(self.data_path, "time_window_consumptions_by_day_type.pickle", self.time_window_consumption_list_dict)
        return judged

    def run(self, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        # Convert to german time and then forget the timezone. The time windows are of the local
        # day; the output carries the UTC time, since the export reads it as UTC.
        utc = pd.Timestamp(timestamp).tz_convert("UTC")
        self.timestamp = utc.tz_convert("Europe/Berlin").tz_localize(None)
        data = {'Consumption': data['Consumption'], 'Time': self.timestamp}
        if not self.first_data_time:
            self.first_data_time = self.timestamp
            self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
            save(self.data_path, "first_data_time.pickle", self.first_data_time)
        logger.debug('energy: '+str(data['Consumption'])+'  '+'time: '+str(self.timestamp))

        if len(self.data_history.index) >= 1 and self.timestamp <= self.data_history.index[-1]: # Discard points that come in wrong order wrt time.
            return
        self.data_history = pd.concat([self.data_history, pd.Series([float(data['Consumption'])], index=[self.timestamp])])
        if len(self.data_history.index) < 2 or self.timestamp.floor('h') != self.data_history.index[-2].floor('h'):
            # Once an hour and not only in stop, so that a crash loses at most the history of the current hour.
            save(self.data_path, "data_history.pickle", self.data_history)
        
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
    
    def stop(self):
        super().stop()
        save(self.data_path, "data_history.pickle", self.data_history)
    
from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="user-profile-operator", git_info_file='git_commit')
