'''
Author: Aiden Pape

Driver function for plotting real time DA diagnostic plots and statistics
'''

from diags import Conventional
import numpy as np
import pandas as pd
from filter_df import filter_df
from make_da_plots import make_base_plots, make_wind_base_plots
from datetime import datetime

def da_base_plots(path, model, date_time, variable, **kwargs):
    
    '''
    Required parameters:
    
    path       : (str) path to the base directory containing the day directories
    model      : (str) which model, 'rtma' or 'rrfs'
    date_time  : (datetime or str) if str in %Y%m%d%H format, datetime of DA run
    variable   : (str) which variable; ps: surface pressure, q: specific humidity, t: temperature, uv: wind, 
                  rw: radial wind, pw: precipitable water, sst: sea surface temperature
    
    Optional parameters:
    
    station_ids: (array str) station_id or ids to filter by
    obs_types  : (array int) bufr ob obs_types to filter by, see link 
               https://emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_2.htm
    use        : (int)   use flag for observation 1=assimilated, 0=not
    p_range    : (float tuple) (min, max) pressure (mb) for including observation in df
    hem        : (string) predetermined geographic extent (GLOBAL, NH, TR, SH, CONUS)
    elv_range  : (float tuple) (min, max) height (m) for including observation in df
    lat_range  : (float tuple) (min, max) latitude (deg N) for including observation in df
    lon_range  : (float tuple) (min, max) latitude (deg E) for including observation in df
    err_range  : (float tuple) (min, max) error std dev for including observation in df (not inversed)
    '''
    
    # Get optional arguments and set defaults
    station_ids = kwargs.get('station_ids', None)
    obs_types = kwargs.get('obs_types', None)
    use = kwargs.get('use', None)
    hem = kwargs.get('hem', None)
    p_range = kwargs.get('p_range', None)
    elv_range = kwargs.get('elv_range', None)
    lat_range = kwargs.get('lat_range', None)
    lon_range = kwargs.get('lon_range', None)
    err_range = kwargs.get('err_range', None)

    # Check a correct model name was given
    model = model.lower()
    if model not in {'rtma', 'rrfs'}:
        raise ValueError(f"'{model}' is not an allowed option, must be 'rtma' or 'rrfs'")
        
    # Convert to string if not already
    if isinstance(date_time, datetime):
        date_time = date_time.strftime('%Y%m%d%H') 
        
    # Get file paths, read, and filter data
    anl_fp = f'diag_conv_{var}_anl.{date_time}.nc4.gz'
    ges_fp = f'diag_conv_{var}_ges.{date_time}.nc4.gz'
    
    anl_diag = Conventional(anl_fp)
    ges_diag = Conventional(ges_fp)
    
    anl_df = anl_diag.get_data()
    ges_df = ges_diag.get_data()
    
    fil_dfs = filter_df([anl_df, ges_df], kwargs)
    
    if(var == 'uv'):
        make_wind_base_plots(fil_dfs, ges_diag.metadata)
    else:
        make_base_plots(fil_dfs, ges_diag.metadata)