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

import util
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kneed
import os
from itertools import chain
import pickle
import datetime
from collections import defaultdict
from . import dynamic_window_determination_user_profile as dwdup

class Operator(util.OperatorBase):
    def __init__(self, device_id, data_path, device_name='das Gerät'):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.device_name = device_name
        self.data_path = data_path

        self.time_window_consumption_list_dict = defaultdict(list)
        self.time_window_consumption_list_dict_anomalies = defaultdict(list)
        self.data_history = pd.Series([], index=[],dtype=object)


        self.window_boundaries_times =  [datetime.time(0, 0), datetime.time(3, 0), datetime.time(6, 0),  datetime.time(9, 0), datetime.time(12, 0), datetime.time(15, 0), 
                                         datetime.time(18, 0), datetime.time(21, 0)]
                                      



        self.consumption_same_time_window = []

        self.current_time_window_start = None
        self.timestamp = None
        self.last_time_window_start = None
        self.last_time_operator_sent_data = pd.Timestamp.now()

        self.time_window_consumption_clustering = {}

        self.clustering_file_path = f'{data_path}/clustering.pickle'
        self.epsilon_file_path = f'{data_path}/epsilon.pickle'
        self.time_window_consumption_list_dict_file_path = f'{data_path}/time_window_consumption_list_dict.pickle'
        self.time_window_consumption_list_dict_anomaly_file_path = f'{data_path}/time_window_consumption_list_dict_anomaly.pickle'
        self.data_history_file = f'{data_path}/data_history.pickle'
        self.window_boundaries_times_file = f'{data_path}/window_boundaries_times.pickle'

        if os.path.exists(self.window_boundaries_times_file):
            with open(self.window_boundaries_times_file, 'rb') as f:
                if os.path.getsize(self.window_boundaries_times_file) > 0:
                    self.window_boundaries_times = pickle.load(f)
        if os.path.exists(self.data_history_file):
            with open(self.data_history_file, 'rb') as f:
                if os.path.getsize(self.data_history_file) > 0:
                    self.data_history = pickle.load(f)
        if os.path.exists(self.time_window_consumption_list_dict_file_path):
            with open(self.time_window_consumption_list_dict_file_path, 'rb') as f:
                if os.path.getsize(self.time_window_consumption_list_dict_file_path) > 0:
                    self.time_window_consumption_list_dict = pickle.load(f)
        if os.path.exists(self.time_window_consumption_list_dict_anomaly_file_path):
            with open(self.time_window_consumption_list_dict_anomaly_file_path, 'rb') as f:
                if os.path.getsize(self.time_window_consumption_list_dict_anomaly_file_path) > 0:
                    self.time_window_consumption_list_dict_anomalies = pickle.load(f)
        

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

    def update_time_window_data(self):
        self.window_boundaries_times = dwdup.window_determination(self.data_history)
        with open(self.window_boundaries_times_file, 'wb') as f:
            pickle.dump(self.window_boundaries_times, f)
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
            self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'].append((self.timestamp, overall_time_window_consumption))
        with open(self.time_window_consumption_list_dict_file_path, 'wb') as f:
            pickle.dump(self.time_window_consumption_list_dict, f)
        return

    def determine_epsilon(self):
        neighbors = NearestNeighbors(n_neighbors=10)
        neighbors_fit = neighbors.fit(np.array([time_window_consumption for _, time_window_consumption in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]]).reshape(-1,1))
        distances, _ = neighbors_fit.kneighbors(np.array([time_window_consumption for _, time_window_consumption in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]]).reshape(-1,1))
        distances = np.sort(distances, axis=0)
        distances_x = distances[:,1]
        kneedle = kneed.KneeLocator(np.linspace(0,1,len(distances_x)), distances_x, S=0.9, curve="convex", direction="increasing")
        epsilon = kneedle.knee_y
        with open(self.epsilon_file_path, 'wb') as f:
            pickle.dump(epsilon, f)
        if epsilon==0 or epsilon==None:
            return 1
        else:
            return epsilon

    def create_clustering(self, epsilon):
        self.time_window_consumption_clustering[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'] = DBSCAN(eps=epsilon, min_samples=7).fit(np.array([time_window_consumption 
                                                                     for _, time_window_consumption in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]]).reshape(-1,1))
        with open(self.clustering_file_path, 'wb') as f:
            pickle.dump(self.time_window_consumption_clustering, f)
        return self.time_window_consumption_clustering[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'].labels_
    
    def test_time_window_consumption(self, clustering_labels):
        anomalous_indices = np.where(clustering_labels==-1)[0]
        quantile = np.quantile([time_window_consumption for _, time_window_consumption in self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:]],0.05)
        anomalous_indices_low = [i for i in anomalous_indices if self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:][i][1] < quantile]
        if len(self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:])-1 in anomalous_indices_low:
            print(f'In letzter Zeit wurde ungewöhnlich wenig verbraucht.')
            self.time_window_consumption_list_dict_anomalies[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'].append(self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-1])
        with open(self.time_window_consumption_list_dict_anomaly_file_path, 'wb') as f:
            pickle.dump(self.time_window_consumption_list_dict_anomalies,f)

        return [self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}'][-14:][i] for i in anomalous_indices_low]
    
    def run(self, data, selector='energy_func'):
        if self.timestamp != None and self.todatetime(data['Time']).tz_localize(None)-self.timestamp < pd.Timedelta(5,'minute'):
            return
        self.timestamp = self.todatetime(data['Time']).tz_localize(None)
        if pd.Timestamp.now() - self.timestamp > pd.Timedelta(157,'d'):
            return
        print('energy: '+str(data['Consumption'])+'  '+'time: '+str(self.timestamp))
        if list(self.data_history.index) and self.timestamp <= self.data_history.index[-1]:
            return
        self.data_history = pd.concat([self.data_history, pd.Series([float(data['Consumption'])], index=[self.timestamp])])
        if self.timestamp.day%30==0 and (self.data_history.index[-1]-self.data_history.index[0] >= pd.Timedelta(10,'d')):
            if self.data_history.index[-2].date()<self.timestamp.date():
                with open(f'{self.data_path}/time_window_consumption_list_dict_{str(self.timestamp.date())}.pickle', 'wb') as f:
                    pickle.dump(self.time_window_consumption_list_dict, f)
                self.update_time_window_data()
                print(self.window_boundaries_times)
        self.current_time_window_start = max(time for time in self.window_boundaries_times if time<=self.timestamp.time())
        if self.consumption_same_time_window == []:
            self.consumption_same_time_window.append(data)
            operator_output = {'value': 0, 'timestamp': str(self.timestamp)}
        elif self.consumption_same_time_window != []:
            self.last_time_window_start = max(time for time in self.window_boundaries_times if time<=self.todatetime(self.consumption_same_time_window[-1]['Time']).tz_localize(None).time())
            if self.current_time_window_start==self.last_time_window_start:
                self.consumption_same_time_window.append(data)
                operator_output = {'value': 0, 'timestamp': str(self.timestamp)}
            else:
                self.consumption_same_time_window.append(data) #!!! Otherwise I lose energy which is consumed between two consecutive windows.
                self.update_time_window_consumption_list_dict()
                with open(self.data_history_file, 'wb') as f:
                    pickle.dump(self.data_history, f)
                if len(self.time_window_consumption_list_dict[f'{str(self.last_time_window_start)}-{str(self.current_time_window_start)}']) >= 10:
                    epsilon = self.determine_epsilon()
                    clustering_labels = self.create_clustering(epsilon)
                    days_with_excessive_consumption_during_this_time_window_of_day = self.test_time_window_consumption(clustering_labels)
                    self.consumption_same_time_window = [data]                 
                    if self.timestamp in list(chain.from_iterable(days_with_excessive_consumption_during_this_time_window_of_day)):
                        operator_output = {'value': 1, 'timestamp': str(self.timestamp)} 
                        return operator_output
                    else:
                        operator_output = {'value': 0, 'timestamp': str(self.timestamp)}
                else:
                    self.consumption_same_time_window = [data] 
                    operator_output = {'value': 0, 'timestamp': str(self.timestamp)}
        
        return operator_output
