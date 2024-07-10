'''
Author: Aiden Pape

Driver function for plotting real time DA diagnostic plots and statistics
'''

from diags import Conventional
import numpy as np
import pandas as pd
from filter_df import filter_df
from da_plots import base_plots, wind_base_plots

def da_base_plots(path, model, variable, station_ids=None, obs_types=None):
    
    '''
    path       : (str) path to the base directory containing the day files
    model      : (str) which model, 'rtma' or 'rrfs'
    variable   : (str) which variable; ps: surface pressure, q: specific humidity, t: temperature, uv: wind, 
                  rw: radial wind, pw: precipitable water, sst: sea surface temperature
    station_ids: (array str) station_id or ids to filter by
    obs_types  : (array int) bufr ob obs_types to filter by, see link 
                  https://emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_2.htm
    '''
    
    # Check a correct model name was given
    model = model.lower()
    if model not in {'rtma', 'rrfs'}:
        raise ValueError(f"'{model}' is not an allowed option, must be 'rtma' or 'rrfs'")