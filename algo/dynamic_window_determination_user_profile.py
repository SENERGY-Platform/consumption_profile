'''
Input for this functionality is a pandas data series which contains the progression of the total consumption (e.g. water consumption or electricity consumption 
of a certain device or device group). The indices of this series are given by the timestamps of the respective values that visualize the consumption progression.
The other input is a pandas timedelta object that depicts the minimum time window length.
'''

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.signal import savgol_filter, find_peaks
import math

def create_daily_ten_min_groups(data_series):
    grouped_by_day = data_series.groupby(by=lambda x: x.floor('d')) 
    # Groups the data series by day

    grouped_ten_min = []
    for group in grouped_by_day:
        grouped_ten_min.append(group[1].groupby(pd.Grouper(freq='10T')))
    
    return grouped_ten_min

def compute_time_window_consumptions(list_of_time_windows):
    consumption_series_list_time_window = []
    for day in list_of_time_windows:
        time_window_data = []
        for window in day:
            if list(window)!=[]:
                time_window_data.append(1000*(window.iloc[-1]-window.iloc[0]))
        consumption_series_list_time_window.append(pd.Series(data=time_window_data, index=[window.index[0].floor('10T') for window in day if list(window)!=[]]))

    for i, series in reversed(list(enumerate(consumption_series_list_time_window))):
        if len(series) != 144:
            del consumption_series_list_time_window[i]
    # The series with less than 144 entries in the list consumption_series_list_ten_min got deleted.

    return consumption_series_list_time_window


def window_determination(data_series):
    resampled_data_series = data_series.resample('T').bfill()
    clean_data_series = resampled_data_series[~resampled_data_series.index.duplicated(keep='first')]
    grouped_ten_min = create_daily_ten_min_groups(clean_data_series)
    # Each object in the list grouped_ten_min contains the data from data_series from a single day grouped into slots of ten minutes.
    grouped_ten_min_list = [[group[1] for group in day] for day in grouped_ten_min]
    
    consumption_series_list_ten_min =  compute_time_window_consumptions(grouped_ten_min_list)                                    
    # Each object in the list consumption_series_list_ten_min contains a pandas series in which for a single day the consumption within the 10-minutes time slots is stored.
    # I.e. eech pandas series in this list has 24*(60/10)=24*6=144 entries if the data is clean. The series with less entries get ignored.

    
    index = [series.index[0].date() for series in consumption_series_list_ten_min] # This just a list which contains the days.

    aux_dict = defaultdict(list)
    for series in consumption_series_list_ten_min:
        for timestamp in list(series.index):
            aux_dict[timestamp.time()].append(series.loc[timestamp])
    df = pd.DataFrame(data=aux_dict, index=index)
    # The pandas dataframe df has the days as indices and the 144 10min time slots in a day as columns. The entries are the consumption from the respective day in the 
    # respective time slot.

    scaled_mean_array = 10000*df.mean(axis=0)
    # We compute the mean along the rows (i.e. days) and scale up by 10000.

    smooth_scaled_mean_array = savgol_filter(scaled_mean_array, 20, 2)
    smoothed_mean_list = smooth_scaled_mean_array.tolist()
    # As a next step we smoothen the arrays and convert them to lists.

    prominence = float(max(smooth_scaled_mean_array))/float(25) # This is kind of random here!?
    peaks_mean, properties_mean = find_peaks(smoothed_mean_list, distance=20, prominence=prominence)
    # Compute the peaks and bases of the peaks.

    # We use these information now to divide the time line into windows (we add the start of the day to the list, i.e. 0).

    window_boundaries_points = sorted(list(set([0]+list(properties_mean['left_bases'])+list(properties_mean['right_bases']))))

    # Now we check if boundary point are too close together (i.e. closer than 0.5 hour):

    points_to_delete_indices = []
    for i, x in enumerate(window_boundaries_points[:-1]):
        if window_boundaries_points[i+1]-x < 3:
            points_to_delete_indices.append(i+1)
    if window_boundaries_points[-1] >= 141:
        points_to_delete_indices.append(len(window_boundaries_points)-1)
    points_to_delete_indices.reverse()    
    for j in points_to_delete_indices:
        del window_boundaries_points[j]

    # Then we convert the boundary points to time objects.

    window_boundaries_times = [(pd.Timestamp.now().floor('d')+i*pd.Timedelta('10T')).time() for i in window_boundaries_points]

    # As a last step we check if the time windows are too long. (Threshold is 2 hours)

    new_window_boundary_times = window_boundaries_times
    for i, time in enumerate(window_boundaries_times[:-1]):
        if pd.Timestamp(str(window_boundaries_times[i+1]))-pd.Timestamp(str(time)) >= pd.Timedelta(2,'h'):
            num_additional_slots = math.ceil((pd.Timestamp(str(window_boundaries_times[i+1]))-pd.Timestamp(str(time)))/pd.Timedelta(2,'h'))
            len_new_slots = (pd.Timestamp(str(window_boundaries_times[i+1]))-pd.Timestamp(str(time)))/num_additional_slots
            additional_slots = []
            for j in range(1,num_additional_slots):
                additional_slots.append((pd.Timestamp(str(time))+j*len_new_slots).time())
            aux = window_boundaries_times.copy()
            aux[i+1:i+1] = additional_slots
            new_window_boundary_times += aux

    if pd.Timestamp(str(window_boundaries_times[0]))-pd.Timestamp(str(window_boundaries_times[-1]))+pd.Timedelta(1,'d') >= pd.Timedelta(2,'h'):
            num_additional_slots = math.ceil((pd.Timestamp(str(window_boundaries_times[0]))-pd.Timestamp(str(window_boundaries_times[-1]))+pd.Timedelta(1,'d'))/pd.Timedelta(2,'h'))
            len_new_slots = (pd.Timestamp(str(window_boundaries_times[0]))-pd.Timestamp(str(window_boundaries_times[-1]))+pd.Timedelta(1,'d'))/num_additional_slots
            additional_slots = []
            for i in range(1,num_additional_slots):
                additional_slots.append((pd.Timestamp(str(window_boundaries_times[-1]))+i*len_new_slots).time())
            aux = window_boundaries_times.copy()
            aux += additional_slots
            new_window_boundary_times += aux

    window_boundaries_times = sorted(list(set(new_window_boundary_times)))

    with open('window_boundaries.txt', 'a') as f:
        for time in window_boundaries_times:
            f.write(str(time)+', ')

        f.write('\n\n')
            



    return window_boundaries_times


'''The input is a list of pd.Timestamp.time() objects and the input is a pandas data series indexed over pd.Timestamps with values given by certain consumption values. 
The first entry of the list is 00:00:00 and the last entry is 24:00:00. The output is given by a list with the following properties: Each entry consists of a collection
of pandas data series which corresponds to the consumption data of one day divided with respect to the list of times given as input.'''
    
def window_division(list_of_timestamps, data_series):
    daily_groups=data_series.groupby(by=lambda x: x.floor('d'))
    grouped_wrt_timestamps = []
    for group in daily_groups:
        data_of_one_day = group[1]
        divided_data_for_one_day = []
        for i, time in enumerate(list_of_timestamps[:-1]):
            divided_data_for_one_day.append(data_of_one_day[group[1].index[0].floor('d')+pd.Timedelta(str(time)):
                                                            group[1].index[0].floor('d')+pd.Timedelta(str(list_of_timestamps[i+1]))])
        grouped_wrt_timestamps.append(divided_data_for_one_day)
    return grouped_wrt_timestamps
    