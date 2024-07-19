'''
Author: Aiden Pape

Read omf/oma data from multiple files to create time series plots

Current not capable of plotting wind time series
'''

from diags import Conventional
import os
import numpy as np
import pandas as pd
from filter_df import filter_df
import matplotlib.pyplot as plt
from datetime import datetime

def plot_time_series(paths, models, var, anl_ges, s_time, f_time, use=None, station_ids=None, obs_types=None, save_plot=False):
    '''
    Get filepaths for RTMA CONUS GSI diag files
    
    Parameters:
    
    paths      : (str list) path to directory containing multiple day directories for each model
    models     : (str list) which models, current options ('rtma_conus', 'rrfs') - Order should match paths
    var        : (str) variable of interest
    anl_ges    : (str) 'anl', 'ges', or 'both if you want to plot analysis or guess
    s_time     : (datetime or str) start date and hour for time series in %Y%m%d%H format
    f_time     : (datetime or str) final date and hour for time series in %Y%m%d%H format
    station_ids: (array str) station_id or ids to filter by
    obs_types  : (array int) bufr ob obs_types to filter by, see link 
                  https://emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_2.htm
                 
    '''
    
    model_options = ['rtma_conus', 'rrfs']
    
    # Check correct argument passed
    if anl_ges not in {'anl', 'ges', 'both'}:
        raise ValueError(f"'{anl_ges}' is not an allowed option, must be 'anl', 'ges', or 'both' ")
    
    #convert times to datetimes if they aren't already
    s_time, f_time = [
        datetime.strptime(t, '%Y%m%d%H') if isinstance(t, str) else t for t in [s_time, f_time]
    ]

    # 
    date_times = pd.date_range(start=s_time, end=f_time, freq='H')
    
    series = {}
    for path, model in zip(paths, models):
        
        #Make sure model is allowed
        if model not in model_options:
            raise ValueError(f"{model} is not an allowed option, must be one or more of {model_options}")
            
        print(f"Reading {model} data...")
            
        # Check need omf/oma time series data
        if anl_ges == 'both' or anl_ges == 'ges':
            ges_fps = get_gsi_fps(path, model, var, 'ges', date_times)
            ges_omfs = get_omfs(ges_fps, var=var, use=use, station_ids=station_ids, obs_types=obs_types)
            #Check if all values are nan
            if(np.all(np.isnan(ges_omfs))): 
                raise ValueError(f"All times for {model} ges are NaN")
            series[f'{model}_ges'] = ges_omfs

        if anl_ges == 'both' or anl_ges == 'anl':
            anl_fps = get_gsi_fps(path, model, var, 'anl', date_times)
            anl_omfs = get_omfs(anl_fps, var=var, use=use, station_ids=station_ids, obs_types=obs_types)
            if(np.all(np.isnan(anl_omfs))): 
                raise ValueError(f"All times for {model} anl are NaN")
            series[f'{model}_anl'] = anl_omfs
            
    print("Making plot..")   
    make_plot(series, date_times, var, save_plot)
    

def get_gsi_fps(path, model, var, anl_ges, date_times):

    '''
    Get filepaths for RTMA CONUS GSI diag files
    
    Parameters:
    
    path       : (str) path to directory containing multiple 
    model      : (str) which model, current options ('rtma', 'rrfs')
    var        : (str) variable of interest
    anl_ges    : (str) 'anl' or 'ges' if you want to plot analysis or guess
    model      : (str) which model, 'rtma_conus' or 'rrfs' (NOT IMPLEMENTED YET_)
    multiday   : (bool) False if path points to folder containing folder 00-23 or True if
                 if points to folder containing folders of different days each day with folders
                 00-23
                 
    Returns:
    
    fps   : (str array) array of strings with filepaths to diag files, chronological
    '''
    if(model=='rtma_conus'):
        model = model.upper()
       
     # Convert DatetimeIndex to array of strings in a specific format
    date_strs = date_times.strftime('%Y%m%d%H').to_numpy()

    day_strs = np.array([s[:8] for s in date_strs])
    hour_strs = np.array([s[-2:] for s in date_strs])
       
    fps = np.array(
        [f'{path}/{model}.{day_str}/{hour_str}/diag_conv_{var}_{anl_ges}.{day_str}{hour_str}.nc4.gz'
        for day_str, hour_str in zip(day_strs, hour_strs)]
    )
    
#     #initialize for 1 day
#     day_dirs = [""]
#     num_days = 1
    
#     if(multiday): #adjust in multiday
#         day_dirs = sorted(os.listdir(path))
#         day_dirs = [fp + '/' for fp in day_dirs]
#         num_days = len(day_dirs)

#     times = np.zeros(24*num_days)
#     fps = [""] * (24*num_days)
#     hours = [f"{i:02d}" for i in range(24)]
    
#     i = 0
#     for day_dir in day_dirs: #loops through days directories, will be "" if one day
#         if(multiday):
#             date_str = day_dir[-9:-1]
#         else:
#             date_str = path[-8:]
            
#         for hour in hours:
#             #get file path and times
#             diag_anl_file = f'diag_conv_{var}_{anl_ges}.{date_str}{hour}.nc4.gz'
#             fp = f'{path}/{day_dir}{hour}/{diag_anl_file}'
#             times[i] = f'{date_str}{hour}'
#             fps[i] = fp
#             i+=1
    
#     times = pd.to_datetime(times, format='%Y%m%d%H')
        
    return fps

def get_omfs(fps, var, use=None, station_ids=None, obs_types=None):
    
    '''
    Read each file and store omf/oma
    
    Parameters:
    
    fps          : (str array) returned by get_gsi_fps, contains filepaths for diag files
    stations_ids : (str list) list containing station_ids to filter for, 
                    default is None so no filtering
    obs_types    : (int list) list containg ints of obs_types to filter for, 
                    default is None so no filtering
                    
    Returns:
    
    hourly_omfs  : (float array) array of omf for each time step, averaged if multiple obs
    '''

    hourly_omf = np.zeros(len(fps)) #Initialize omf array

    for i, fp in enumerate(fps):
        try:
            #Open file and filter dataframe
            df = Conventional(fp).get_data()
            
            omf_values = filter_df([df], use=use, station_ids=station_ids, obs_types=obs_types)[0]['omf_adjusted']
            
            if(len(omf_values)==0):
                print('Filter combination yields no results')
                
            #Convert to fahrenheit
            if(var=='t'):
                omf_values = omf_values * 1.8
                
            hourly_omf[i] = omf_values.mean()
            
        except FileNotFoundError:
#             print(f"FileNotFound: {fp}")
            hourly_omf[i]= None
        except Exception as e:
            print(f"Unknown error occurred: {e}")
            raise
            
    return hourly_omf

def make_plot(series, times, var, save_plot):
    
    '''
    Make time series plot of omf/oma, can plot both omf and oma of the same time and var
    
    Parameters:
    
    series: (list of float arrays) each returned by get_omfs, omf for each hour on a time domain, max 6
    times : (datetime array) returned by get_gsi_fps, corresponding datetimes for each hour
    var   : (str) variable of the series
    '''
    
    # Variable name dict
    units_mapping = {
        't': ['Temperature', 'Degrees Fahrenheit'],
        'ps': ['Surface Pressure', 'Pascals'],
        'q': ['Specific Humidity', 'G Water Vapor per KG of Air']
    }
    
    # Initialize var_units based on metadata
    var_units = units_mapping.get(var)[1]
    var_name = units_mapping.get(var)[0]

    # Create the plot
    fig = plt.figure(figsize=(10, 5))
    
    # Define a list of colors to cycle through
    colors = ['b', 'g', 'c', 'm', 'y', 'k']
    
    color_ind = 0
    max_val = 0
    
    plt_name = ""
    
    for key, data in series.items():
        
        plt_name = plt_name + key
        
        plt.plot(times, data, marker='o', linestyle='-', color=colors[color_ind], label=key)
        color_ind = (color_ind + 1) % len(colors)
        
        # Highlight NaN points
        nan_indices = np.isnan(data)
        plt.scatter(times[nan_indices], np.zeros(sum(nan_indices)), color='red')
        #ToDo, add NaN label on legend but only once
        
        max_val = max((np.nanmax(np.abs(data)) * 1.1), max_val)
        
    # Plot a horizontal line at zero
    plt.axhline(y=0, color='r', linestyle='--')

    # Set y-axis limits to have zero line in the middle
    plt.ylim(-max_val, max_val)

    # plt.xlabel('Date')
    plt.ylabel(var_units)
    plt.title(f'{var_name} DA Time Series Plot')
    plt.grid(True)
    plt.legend()

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    if(save_plot):
        os.makedirs("plots", exist_ok=True)
        fig_name = f'{var}_{plt_name}_time_series'
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(fig)
        print(f'Figure saved as: {fig_name}')
    else:
        plt.show()
    