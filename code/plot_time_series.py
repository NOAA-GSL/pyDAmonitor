'''
Author: Aiden Pape

Read omf/oma data from multiple files to create time series plots
'''

from diags import Conventional
import os
import numpy as np
import pandas as pd
from filter_df import filter_df
import matplotlib.pyplot as plt

def get_gsi_fps(path, var, anl_or_ges, multiday=False):

    '''
    Get filepaths for RTMA CONUS GSI diag files
    
    Parameters:
    
    path       : (str) path to directory containing
    var        : (str) variable of interest
    anl_or_ges : (str) 'anl' or 'ges' if you want to plot analysis or guess
    multiday   : (bool) False if path points to folder containing folder 00-23 or True if
                 if points to folder containing folders of different days each day with folders
                 00-23
                 
    Returns:
    
    fps   : (str array) array of strings with filepaths to diag files, chronological
    times : (datetime array) array containing corresponding datetimes for each file
    '''
    
    if anl_or_ges not in {'anl', 'ges'}:
        raise ValueError(f"'{anl_or_ges}' is not an allowed option, must be 'anl' or 'ges'")
    
    #initialize for 1 day
    day_dirs = [""]
    num_days = 1
    
    if(multiday): #adjust in multiday
        day_dirs = sorted(os.listdir(path))
        day_dirs = [fp + '/' for fp in day_dirs]
        num_days = len(day_dirs)

    times = np.zeros(24*num_days)
    fps = [""] * (24*num_days)
    hours = [f"{i:02d}" for i in range(24)]
    
    i = 0
    for day_dir in day_dirs: #loops through days directories, will be "" if one day
        if(multiday):
            date_str = day_dir[-9:-1]
        else:
            date_str = path[-8:]
            
        for hour in hours:
            #get file path and times
            diag_anl_file = f'diag_conv_{var}_{anl_or_ges}.{date_str}{hour}.nc4.gz'
            fp = f'{path}/{day_dir}{hour}/{diag_anl_file}'
            times[i] = f'{date_str}{hour}'
            fps[i] = fp
            i+=1
    
    times = pd.to_datetime(times, format='%Y%m%d%H')
        
    return fps, times

def get_omfs(fps, station_ids=None, obs_types=None):
    
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
            omf_values = filter_df([df], station_ids=station_ids, obs_types=obs_types)['omf_adjusted']
            
            if(len(omf_values)==0):
                print('Filter combination yields no results')
                
            hourly_omf[i] = omf_values.mean()
            
        except FileNotFoundError:
    #         print(f"FileNotFound: {hour}/{diag_anl_file}")
            hourly_omf[i]= None
        except Exception as e:
            print(f"Unknown error occurred: {e}")
            raise
            
    return hourly_omf

def make_plot(hourly_omf, times):
    
    '''
    Make time series plot of omf/oma
    
    Parameters:
    
    hourly_omf : (float array) returned by get_omfs, omf for each hour on a time domain
    times      : (datetime array) returned by get_gsi_fps, corresponding datetimes for each hour
    '''

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(times, hourly_omf, marker='o', linestyle='-', color='b', label='OmF')

    # Plot a horizontal line at zero
    plt.axhline(y=0, color='r', linestyle='--', label='Zero')

    # Highlight NaN points
    nan_indices = np.isnan(hourly_omf)
    plt.scatter(times[nan_indices], np.zeros(sum(nan_indices)), color='red', 
                label='NaN Points')

    # Set y-axis limits to have zero line in the middle
    max_val = np.nanmax(np.abs(hourly_omf)) * 1.1
    plt.ylim(-max_val, max_val)

    # plt.xlabel('Date')
    plt.ylabel('OmF')
    plt.title('OmF Time Series Plot')
    plt.grid(True)
    plt.legend()

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    plt.show()