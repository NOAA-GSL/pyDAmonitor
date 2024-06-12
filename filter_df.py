# Code taken from PyGSI and altered

import numpy as np

def filter_df(
    df,
    obs_types,
    use=None,
    hem=None,
    p_range = None,
    elv_range = None,
    lat_range = None,
    lon_range = None,
    err_range = None,
    in_place = False
):
    """
    Create the bool array mask to filter df by. Considers obs filtered by the use flag,
    max/min pressure, latitude, longitude, and errors along with filtering by bufr
    obs_type (e.g., 187 or 287).

        df       : a pandas df returned by PyGSI.Conventional.get_data() (NOTE) with some small changes
                    being that indices are reset and use flag is 0 for not assimilated not -1

    Values needed from df:
        obs_type     : (array int) bufr ob obs_types from netCDF file <Observation_Type>
        errorinv : (array float) from netCDF file <Errinv_Final>
        lat      : (array float) from netCDF file <Latitude>
        lon      : (array float) from netCDF file <Longitude>
        pressure : (array float) from netCDF file <Pressure>

     Args:
        obs_types    : (array int) bufr ob obs_types to filter by, see link 
                   https://emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_2.htm
        use      : (int)   use flag for observation 1=assimilated
        p_range  : (float tuple) (min, max) pressure (mb) for including observation in df
        elv_range: (float tuple) (min, max) height (m) for including observation in df
        lat_range: (float tuple) (min, max) latitude (deg N) for including observation in df
        lon_range: (float tuple) (min, max) latitude (deg E) for including observation in df
        err_range: (float tuple) (min, max) error std dev for including observation in df (not inversed)
        inPlace  : (bool) True to filter current df, False to return reference to a new filtered df

    Returns:
        Returns new df is in_place is false and edits passed df is in_place is true 
    """

    #make a copy if this filter is not inplace, else leave it to alter passed df
    if not in_place:
        df = df.copy()

    use_flag = df['Analysis_Use_Flag']
    obs_type = df['Observation_Type']
    errorinv = df['errinv_final']
    lat =  df['latitude']
    lon = df['longitude']
    pressure = df['Pressure']
    elevation = df['station_elevation']

    # if hem is provided, override the lat/lon min/maxes
    if hem is not None:

        if (hem == "GLOBAL"):
            lat_max = 90.0
            lat_min = -90.0
            lon_max = 360.0
            lon_min = 0.0
        elif (hem == "NH"):
            lat_max = -30.0
            lat_min = -90.0
            lon_max = 360.0
            lon_min = 0.0
        elif (hem == "TR"):
            lat_max = 30.0
            lat_min = -30.0
            lon_max = 360.0
            lon_min = 0.0
        elif (hem == "SH"):
            lat_max = 90.0
            lat_min = 30.0
            lon_max = 360.0
            lon_min = 0.0
        elif (hem == "CONUS"):
            lat_max = 50.0
            lat_min = 27.0
            lon_max = 295.0
            lon_min = 235.0
        else:
            msg = 'hemispheres must be: GLOBAL, NH, TR, SH, CONUS, or None'
            raise ValueError(msg)

    #filter for the desired obs types
    mask = obs_type == obs_types[0]  # initialize mask
    for cd in obs_types:
        mask = np.logical_or(mask, obs_type == cd)  # loop over all obs_types provided

    # filter for desired use flag if provided
    if use is not None:
        mask = np.logical_and(mask, use == use_flag)

    # filter by pressure if provided
    if p_range is not None:
        mask = np.logical_and(mask, np.logical_and(pressure >= p_range[0], pressure <= p_range[1]))

    # filter by elevation if provided
    if elv_range is not None:
        mask = np.logical_and(mask, np.logical_and(elevation >= elv_range[0], elevation <= elv_range[1]))

    # filter bound by lats if provided
    if lat_range is not None:
        mask = np.logical_and(mask, np.logical_and(lat >= lat_range[0], lat <= lat_range[1]))

    # filter bound by lons if provided
    if lon_range is not None:
        mask = np.logical_and(mask, np.logical_and(lon >= lon_range[0], lon <= lon_range[1]))

    #set error mins and maxs and filter for desired error values if provided
    if err_range is not None:
        error_min_inv = 1.0 / err_range[0]
        error_max_inv = 1.0 / err_range[1]
        mask = np.logical_and(mask, np.logical_and(errorinv <= error_min_inv, errorinv >= error_max_inv))

    #filter dataframe, keeping only desired obs
    df.drop(df[~mask].index, inplace = True)

    return df