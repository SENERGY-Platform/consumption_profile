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

from operator_lib.util import OperatorBase, logger, InitPhase
from operator_lib.util.persistence import save, load
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kneed
import os
from itertools import chain
import datetime
from collections import defaultdict
from algo import dynamic_window_determination_user_profile as dwdup

from operator_lib.util import Config
class CustomConfig(Config):
    data_path = "/opt/data"
    init_phase_length: float = 2
    init_phase_level: str = "d"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 2
        
        if self.init_phase_level == '':
            self.init_phase_level = 'd'

class Operator(OperatorBase):
    configType = CustomConfig

    def init(self,  *args, **kwargs):
        super().init(*args, **kwargs)
        self.data_path = self.config.data_path
        self.first_data_time = load(self.config.data_path, "first_data_time.pickle")

        self.time_window_data_just_updated = None



        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)

        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        value = {
                    "value": 0,
                    "timestamp": "",
                    "message": "",
                    "last_consumptions": "",
                    "time_window": ""
        }
        self.init_phase_handler.send_first_init_msg(value)
        
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.consumption_same_time_window = []

        self.time_window_consumption_clustering = {}
        
        self.window_boundaries_times = load(self.data_path, 'window_boundaries_times.pickle', default=[datetime.time(0, 0), datetime.time(3, 0), datetime.time(6, 0),  datetime.time(9, 0), datetime.time(12, 0), datetime.time(15, 0), 
                                         datetime.time(18, 0), datetime.time(21, 0)])
        self.data_history = load(self.data_path, "data_history.pickle", default=pd.Series([], index=[],dtype=object))
        self.time_window_consumption_list_dict = load(self.data_path, "time_window_consumption_list_dict.pickle", default=defaultdict(list))
        self.time_window_consumption_list_dict_anomalies = load(self.data_path, "time_window_consumption_list_dict_anomaly.pickle", default=defaultdict(list))
        

    def todatetime(self, timestamp):
        if str(timestamp).isdigit():
            if len(str(timestamp))==13:
                return pd.to_datetime(int(timestamp), unit='ms')
            elif len(str(timestamp))==19:
                return pd.to_datetime(int(timestamp), unit='ns')
        else:
            return pd.to_datetime(timestamp)

    def create_new_time_window_consumption_list_dict(self):
        self.time_window_consumption_list_dict = defaultdict(list)
        list_of_time_windows = dwdup.window_division(self.window_boundaries_times, self.data_history[(self.timestamp-pd.Timedelta(14,'d')).floor('d'):])
        for day in list_of_time_windows:
            for window_name, window in day:
                if list(window)!=[]:
                    overall_time_window_consumption = 1000*(window.iloc[-1]-window.iloc[0])
                    if overall_time_window_consumption>=0:
                        self.time_window_consumption_list_dict[window_name].append((window.index[-1], overall_time_window_consumption))
        save(self.data_path, "time_window_consumption_list_dict.pickle", self.time_window_consumption_list_dict)

    def update_time_window_data(self):
        self.window_boundaries_times = dwdup.window_determination(self.data_history, self.window_boundaries_times)
        save(self.data_path, 'window_boundaries_times.pickle', self.window_boundaries_times)
        self.create_new_time_window_consumption_list_dict()

    def update_time_window_consumption_list_dict(self):
        if (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > 3*pd.Timedelta(1,'h')) or (
           (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) < 0*pd.Timedelta(1,'h')) and (
            pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > -21*pd.Timedelta(1,'h')
           ) 
        ):
            return
        time_window_consumption_max = float(self.consumption_same_time_window[-1]['Consumption'])
        time_window_consumption_min = float(self.consumption_same_time_window[0]['Consumption'])
        overall_time_window_consumption = 1000*(time_window_consumption_max-time_window_consumption_min)
        if np.isnan(overall_time_window_consumption)==False and overall_time_window_consumption >= 0:
            if self.time_window_data_just_updated == False:
                self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'].append((self.timestamp, overall_time_window_consumption))
            elif self.time_window_data_just_updated == True:
                # If in the current time window the time_window_data was updated there is already an entry for the current time window from the current day. The consumption
                # that is stored there depicts the consumption from the window until the point of time_window_data updating. The rest of the consumption during the window has just
                # to be added.
                last_entry_for_current_time_window = self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-1]
                self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-1] = (self.timestamp, last_entry_for_current_time_window[1]+overall_time_window_consumption)
                self.time_window_data_just_updated = False
        save(self.data_path, "time_window_consumption_list_dict.pickle", self.time_window_consumption_list_dict)
        return

    def determine_epsilon(self):
        neighbors = NearestNeighbors(n_neighbors=10)
        neighbors_fit = neighbors.fit(np.array([time_window_consumption for _, time_window_consumption in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]]).reshape(-1,1))
        distances, _ = neighbors_fit.kneighbors(np.array([time_window_consumption for _, time_window_consumption in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]]).reshape(-1,1))
        distances = np.sort(distances, axis=0)
        distances_x = distances[:,1]
        kneedle = kneed.KneeLocator(np.linspace(0,1,len(distances_x)), distances_x, S=0.9, curve="convex", direction="increasing")
        epsilon = kneedle.knee_y
        if epsilon==0 or epsilon==None:
            return 1
        else:
            return epsilon

    def create_clustering(self, epsilon):
        self.time_window_consumption_clustering[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'] = DBSCAN(eps=epsilon, min_samples=7).fit(np.array([time_window_consumption 
                                                                     for _, time_window_consumption in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]]).reshape(-1,1))
        return self.time_window_consumption_clustering[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'].labels_
    
    def test_time_window_consumption(self, clustering_labels):
        anomalous_indices = np.where(clustering_labels==-1)[0]
        quantile = np.quantile([time_window_consumption for _, time_window_consumption in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]],0.05)
        anomalous_indices_low = [i for i in anomalous_indices if self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:][i][1] < quantile]
        if len(self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:])-1 in anomalous_indices_low:
            logger.warning(f'In letzter Zeit wurde ungewöhnlich wenig verbraucht.')
            print(self.time_window_consumption_list_dict)
            self.time_window_consumption_list_dict_anomalies[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'].append(self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-1])
        save(self.data_path, "time_window_consumption_list_dict_anomaly.pickle", self.time_window_consumption_list_dict_anomalies)

        return [self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:][i] for i in anomalous_indices_low]
    
    def run(self, data, selector='energy_func',device_id=''):
        self.timestamp = self.todatetime(data['Time']).tz_localize(None)
        if not self.first_data_time:
            self.first_data_time = self.timestamp
            self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
            save(self.data_path, "first_data_time.pickle", self.first_data_time)
        logger.debug('energy: '+str(data['Consumption'])+'  '+'time: '+str(self.timestamp))

        if len(self.data_history.index) >= 2 and self.timestamp <= self.data_history.index[-2]: # Discard points that come in wrong order wrt time.
            return
        self.data_history = pd.concat([self.data_history, pd.Series([float(data['Consumption'])], index=[self.timestamp])])
        
        operator_is_init = self.init_phase_handler.operator_is_in_init_phase(self.timestamp)
        
        
        
        init_value = {
                    "value": 0,
                    "timestamp": str(self.timestamp),
                    "message": "",
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
            logger.debug(self.window_boundaries_times)
            self.consumption_same_time_window = []
        self.current_time_window_start = max(time for time in self.window_boundaries_times if time<=self.timestamp.time())
        if self.consumption_same_time_window == []:
            self.consumption_same_time_window.append(data)
        elif self.consumption_same_time_window != []:
            self.last_time_window_start = max(time for time in self.window_boundaries_times if time<=self.todatetime(self.consumption_same_time_window[-1]['Time']).tz_localize(None).time())
            if self.current_time_window_start==self.last_time_window_start:
                self.consumption_same_time_window.append(data)
            elif (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > 3*pd.Timedelta(1,'h')) or (  # If the gap between the current and last window boundary is too large 
                 (pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) < 0*pd.Timedelta(1,'h')) and ( # (because for example there were data points missing in the stream) we ignore the current consumption
                  pd.Timestamp(str(self.current_time_window_start))-pd.Timestamp(str(self.last_time_window_start)) > -21*pd.Timedelta(1,'h'))):   # and start again.
                self.consumption_same_time_window = [data] 
                return
            else:
                self.consumption_same_time_window.append(data) #!!! Otherwise I lose energy which is consumed between two consecutive windows.
                self.update_time_window_consumption_list_dict()
                epsilon = self.determine_epsilon()
                clustering_labels = self.create_clustering(epsilon)
                days_with_excessive_consumption_during_this_time_window_of_day = self.test_time_window_consumption(clustering_labels)
                df_cons_last_14_days = self.create_df_cons_last_14_days()
                self.consumption_same_time_window = [data]                 
                if self.timestamp in list(chain.from_iterable(days_with_excessive_consumption_during_this_time_window_of_day)):
                    operator_output = self.create_output(1, self.timestamp, df_cons_last_14_days)
                    return operator_output
                else:
                    operator_output = self.create_output(0, self.timestamp, df_cons_last_14_days)
                    return operator_output
    
    def create_df_cons_last_14_days(self):
        days = [timestamp.date() for timestamp, _ in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]]
        print(f"Days: {days}")
        time_window_consumptions = [consumption for _, consumption in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]]
        print(f"Time Window Consumptions: {time_window_consumptions}")
        df = pd.DataFrame(time_window_consumptions, index=days)
        print(f"Df: {df}")
        return df.reset_index(inplace=False).to_json(orient="values")
    
    def create_output(self, anomaly, timestamp, df_cons_last_14_days):
        if anomaly == 0:
            message = ""
        elif anomaly == 1:
            message = f"In der Zeit zwischen {str(self.last_time_window_start)} und {str(self.current_time_window_start)} wurde ungewöhnlich wenig verbraucht."
        return {
                    "value": anomaly,
                    "timestamp": str(timestamp),
                    "message": message,
                    "last_consumptions": df_cons_last_14_days,
                    "time_window": f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'
        }
    
    def stop(self):
        super().stop()
        save(self.data_path, "data_history.pickle", self.data_history)
    
from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="user-profile-operator", git_info_file='git_commit')
