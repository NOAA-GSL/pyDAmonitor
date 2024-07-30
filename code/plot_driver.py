'''
Author: Aiden Pape

Driver function for plotting real time DA diagnostic plots and statistics
'''

from GSI_diags import Conventional
from filter_df import filter_df
from make_da_plots import make_base_plots, make_wind_base_plots
from datetime import datetime
import argparse
import yaml

def da_base_plots(path, model, date_time, var, **kwargs):
    
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
    # anl_fp = f'{path}/diag_conv_{var}_anl.{date_time}.nc4.gz'
    # ges_fp = f'{path}/diag_conv_{var}_ges.{date_time}.nc4.gz'
    # for non zipped files, idk what to use atm
    
    print("Reading and filtering data...")
    anl_fp = f'{path}/diag_conv_{var}_anl.{date_time}.nc4'
    ges_fp = f'{path}/diag_conv_{var}_ges.{date_time}.nc4'
    
    anl_diag = Conventional(anl_fp)
    ges_diag = Conventional(ges_fp)
    
    anl_df = anl_diag.get_data()
    ges_df = ges_diag.get_data()
    
    fil_dfs = filter_df([anl_df, ges_df], station_ids=station_ids, obs_types=obs_types, use=use, hem=hem, p_range=p_range,
                        elv_range=elv_range, lat_range=lat_range, lon_range=lon_range, err_range=err_range)
    
    print("Data read successfully\n")
    
    if(var == 'uv'):
        # make_wind_base_plots(fil_dfs, ges_diag.metadata, save_plots=True)
        make_wind_base_plots(fil_dfs, ges_diag.metadata, save_plots=True) #for testing purposes
    else:
        # make_base_plots(fil_dfs, ges_diag.metadata, save_plots=True)
        make_base_plots(fil_dfs, ges_diag.metadata, save_plots=True)
        
def main():
    
    parser = argparse.ArgumentParser(description="Run DA base plots.")

    # Required arguments (for command line arguments, not actually required bc we can use config files)
    parser.add_argument('--path', type=str, help="Path to the base directory containing the day directories.")
    parser.add_argument('--model', type=str, choices=['rtma', 'rrfs'], help="Model to use ('rtma' or 'rrfs').")
    parser.add_argument('--date_time', type=str, help="Datetime of DA run in '%Y%m%d%H' format.")
    parser.add_argument('--var', type=str, help="Variable to plot, options: (ps, q, t, uv, rw, pw, sst).")

    # Optional arguments (for command line arguments)
    parser.add_argument('--station_ids', nargs='+', help="Station ID(s) to filter by.")
    parser.add_argument('--obs_types', nargs='+', type=int, help="Observation types to filter by.")
    parser.add_argument('--use', type=int, help="Use flag for observation (1=assimilated, 0=not).")
    parser.add_argument('--hem', type=str, help="Predetermined geographic extent (GLOBAL, NH, TR, SH, CONUS).")
    parser.add_argument('--p_range', nargs=2, type=float, help="Pressure range (min max) for including observation in df.")
    parser.add_argument('--elv_range', nargs=2, type=float, help="Elevation range (min max) for including observation in df.")
    parser.add_argument('--lat_range', nargs=2, type=float, help="Latitude range (min max) for including observation in df.")
    parser.add_argument('--lon_range', nargs=2, type=float, help="Longitude range (min max) for including observation in df.")
    parser.add_argument('--err_range', nargs=2, type=float, help="Error range (min max) for including observation in df.")
    parser.add_argument('--config_file', type=str, help="Path to the YAML configuration file.")

    args = parser.parse_args()
    
    #check if a config file was passed
    if args.config_file:
        
        print("Arguments passed with configuration file")
        
        #Open config file
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)
            
        # get kwargs
        kwargs = {
            'station_ids': config.get('station_ids'),
            'obs_types': config.get('obs_types'),
            'use': config.get('use'),
            'hem': config.get('hem'),
            'p_range': tuple(config.get('p_range')) if config.get('p_range') else None,
            'elv_range': tuple(config.get('elv_range')) if config.get('elv_range') else None,
            'lat_range': tuple(config.get('lat_range')) if config.get('lat_range') else None,
            'lon_range': tuple(config.get('lon_range')) if config.get('lon_range') else None,
            'err_range': tuple(config.get('err_range')) if config.get('err_range') else None
        }
        
        # remove none args 
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        #debugging
        # for key, value in config.items():
        #     print(f"{key}: {value}")    
        
        #call function
        da_base_plots(config['path'], config['model'], config['date_time'], config['var'], **kwargs)
        
    # no config file so we will use command like args
    else:
        
        print("No configuration file passed, reading arguments from command line")
        # get kwargs
        kwargs = {
            'station_ids': args.station_ids,
            'obs_types': args.obs_types,
            'use': args.use,
            'hem': args.hem,
            'p_range': tuple(args.p_range) if args.p_range else None,
            'elv_range': tuple(args.elv_range) if args.elv_range else None,
            'lat_range': tuple(args.lat_range) if args.lat_range else None,
            'lon_range': tuple(args.lon_range) if args.lon_range else None,
            'err_range': tuple(args.err_range) if args.err_range else None
        }

        # Remove None args
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # call function
        da_base_plots(args.path, args.model, args.date_time, args.var, **kwargs)

if __name__ == "__main__":
    main()