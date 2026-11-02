'''
Input for this functionality is a pandas data series which contains the progression of the total consumption (e.g. water consumption or electricity consumption 
of a certain device or device group). The indices of this series are given by the timestamps of the respective values that visualize the consumption progression.
The other input is a pandas timedelta object that depicts the minimum time window length.
'''

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import math

WEEKDAY = 'weekday'
WEEKEND = 'weekend'
DAY_TYPES = (WEEKDAY, WEEKEND)
# Weekdays and weekends have time windows of their own and are only compared with days of the same type.

def type_of_day(day):
    return WEEKEND if day.weekday() >= 5 else WEEKDAY

def compute_ten_min_consumptions(data_series):
    slots = data_series.index.floor('10T')
    first = data_series.groupby(slots).head(1)
    last = data_series.groupby(slots).tail(1)
    consumptions = pd.Series(1000*(last.to_numpy()-first.to_numpy()), index=first.index.floor('10T'))
    # The consumption within a ten-minute slot is the difference between its last and its first value.

    consumptions = consumptions[consumptions.groupby(consumptions.index.floor('d')).transform('size') == 144]
    # Days with less than 144 slots are left out.

    return pd.DataFrame({'day': consumptions.index.date, 'slot': consumptions.index.time, 'consumption': consumptions.to_numpy()}).pivot(
        index='day', columns='slot', values='consumption')


def window_determination(data_series, window_boundaries_times, day_type=None):
    try:
        resampled_data_series = data_series.resample('T').bfill()
    except:
        return window_boundaries_times
    clean_data_series = resampled_data_series[~resampled_data_series.index.duplicated(keep='first')]
    df = compute_ten_min_consumptions(clean_data_series)
    # The pandas dataframe df has the days as indices and the 144 10min time slots in a day as columns. The entries are the consumption from the respective day in the
    # respective time slot.

    if day_type is not None:
        df = df[[type_of_day(day) == day_type for day in df.index]]
    # Only the days of the given type count. They are selected only here, since the series without the other days would have
    # gaps that the resampling fills with consumption.
    if df.empty:
        return window_boundaries_times

    df[df<0]=np.nan
    # Here we discard negative values in the dataframe. Negative values do not make sense and can only occure through wrong data.

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

    # As a last step we check if the time windows are too long. (Threshold is 3 hours)

    new_window_boundary_times = window_boundaries_times
    for i, time in enumerate(window_boundaries_times[:-1]):
        if pd.Timestamp(str(window_boundaries_times[i+1]))-pd.Timestamp(str(time)) > pd.Timedelta(3,'h'):
            num_additional_slots = math.ceil((pd.Timestamp(str(window_boundaries_times[i+1]))-pd.Timestamp(str(time)))/pd.Timedelta(3,'h'))
            len_new_slots = (pd.Timestamp(str(window_boundaries_times[i+1]))-pd.Timestamp(str(time)))/num_additional_slots
            additional_slots = []
            for j in range(1,num_additional_slots):
                additional_slots.append((pd.Timestamp(str(time))+j*len_new_slots).time())
            aux = window_boundaries_times.copy()
            aux[i+1:i+1] = additional_slots
            new_window_boundary_times += aux

    if pd.Timestamp(str(window_boundaries_times[0]))-pd.Timestamp(str(window_boundaries_times[-1]))+pd.Timedelta(1,'d') > pd.Timedelta(3,'h'):
            num_additional_slots = math.ceil((pd.Timestamp(str(window_boundaries_times[0]))-pd.Timestamp(str(window_boundaries_times[-1]))+pd.Timedelta(1,'d'))/pd.Timedelta(3,'h'))
            len_new_slots = (pd.Timestamp(str(window_boundaries_times[0]))-pd.Timestamp(str(window_boundaries_times[-1]))+pd.Timedelta(1,'d'))/num_additional_slots
            additional_slots = []
            for i in range(1,num_additional_slots):
                additional_slots.append((pd.Timestamp(str(window_boundaries_times[-1]))+i*len_new_slots).time())
            aux = window_boundaries_times.copy()
            aux += additional_slots
            new_window_boundary_times += aux

    window_boundaries_times = sorted(list(set(new_window_boundary_times)))

    return window_boundaries_times


'''The input is a list of pd.Timestamp.time() objects and the input is a pandas data series indexed over pd.Timestamps with values given by certain consumption values. 
The first entry of the list is 00:00:00 and the last entry is 24:00:00. The output is given by a list with the following properties: Each entry consists of the day and a collection
of pandas data series which corresponds to the consumption data of that day divided with respect to the list of times given as input.'''
    
def window_division(list_of_timestamps, data_series):
    daily_groups=data_series.groupby(by=lambda x: x.floor('d'))
    grouped_wrt_timestamps = []
    for group in daily_groups:
        data_of_one_day = group[1]
        divided_data_for_one_day = []
        for i, time in enumerate(list_of_timestamps[:-1]):
            aux_series_start = data_of_one_day[group[1].index[0].floor('d')+pd.Timedelta(str(time)):
                                                            group[1].index[0].floor('d')+pd.Timedelta(str(list_of_timestamps[i+1]))]
            aux_series_end = data_of_one_day[group[1].index[0].floor('d')+pd.Timedelta(str(list_of_timestamps[i+1])):]
            if len(aux_series_start) > 0 and len(aux_series_end) > 0 and aux_series_start.index[-1]==aux_series_end.index[0]:
                divided_data_for_one_day.append((f'{str(time)}-{list_of_timestamps[i+1]}', pd.concat([aux_series_start, aux_series_end.iloc[1:2]])))
            elif len(aux_series_start) > 0 and len(aux_series_end) > 0 and aux_series_start.index[-1]!=aux_series_end.index[0]:
                divided_data_for_one_day.append((f'{str(time)}-{list_of_timestamps[i+1]}', pd.concat([aux_series_start, aux_series_end.iloc[0:1]])))
            elif len(aux_series_start) == 0 and len(aux_series_end) > 0:
                divided_data_for_one_day.append((f'{str(time)}-{list_of_timestamps[i+1]}', pd.Series([], dtype=object)))
            elif len(aux_series_start) > 0 and len(aux_series_end) == 0:
                divided_data_for_one_day.append((f'{str(time)}-{list_of_timestamps[i+1]}', aux_series_start))
            else:
                divided_data_for_one_day.append((f'{str(time)}-{list_of_timestamps[i+1]}', pd.Series([], dtype=object)))
        last_window = data_of_one_day[group[1].index[0].floor('d')+pd.Timedelta(str(list_of_timestamps[i+1])):]
        if len(last_window) > 0:
            # Like every other window, the last one ends at the first value after it, here the first one of the next day.
            last_window = pd.concat([last_window, data_series[group[1].index[0].floor('d')+pd.Timedelta(1,'d'):].iloc[0:1]])
        divided_data_for_one_day.append((f'{list_of_timestamps[-1]}-00:00:00', last_window))
        grouped_wrt_timestamps.append((group[0], divided_data_for_one_day))
    #print(grouped_wrt_timestamps) 
    return grouped_wrt_timestamps
    