# Code taken from PyGSI and altered

import numpy as np

def filter_df(dfs, **kwargs):
    """
    Create the bool array mask to filter df by. Considers obs filtered by the use flag,
    max/min pressure, latitude, longitude, and errors along with filtering by bufr
    obs_type (e.g., 187 or 287).

        dfs         : a list of pandas dfs, each returned by inputing a netCDF file into
                        diags.Conventional.get_data()

    Values needed from df:
        station_ids: (array str) id of station observation was taken at
        use_flag   : (array int) 1 if assimilated, 0 if not
        obs_type   : (array int) bufr observation obs_types <Observation_Type>
        errorinv   : (array float) observation <Errinv_Final>
        lat        : (array float) observation <Latitude>
        lon        : (array float) observation <Longitude>
        pressure   : (array float) observation <Pressure>
        elevation  : (array float) ovservation elevation

     Args:
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
        inPlace    : (bool) True to filter current df, False to return reference to a new filtered df

    Returns:
        Returns new dfs if in_place is false and edits passed dfs is in_place is true, returns a list in the same order they were passed
        
    """
    
    # Default values for parameters
    station_ids = kwargs.get('station_ids', None)
    obs_types = kwargs.get('obs_types', None)
    use = kwargs.get('use', None)
    hem = kwargs.get('hem', None)
    p_range = kwargs.get('p_range', None)
    elv_range = kwargs.get('elv_range', None)
    lat_range = kwargs.get('lat_range', None)
    lon_range = kwargs.get('lon_range', None)
    err_range = kwargs.get('err_range', None)
    in_place = kwargs.get('in_place', False)
    
    fil_dfs = []
    
    for df in dfs:

        #make a copy if this filter is not inplace, else leave it to alter passed df
        if not in_place:
            df = df.copy()

        station_id = df['station_id']
        use_flag = df['analysis_use_flag']
        obs_type = df['observation_type']
        error = df['error']
        lat =  df['latitude']
        lon = df['longitude']
        pressure = df['pressure']
        elevation = df['station_elevation']

        # if hem is provided, override the lat/lon min/maxes
        if hem is not None:

            if (hem == "GLOBAL"):
                lat_range = (-90.0, 90.0)
                lon_range = (0.0, 360.0)
            elif (hem == "NH"):
                lat_range = (-90.0, -30.0)
                lon_range = (0.0, 360.0)
            elif (hem == "TR"):
                lat_range = (-30.0, 30.0)
                lon_range = (0.0, 360.0)
            elif (hem == "SH"):
                lat_range = (-90.0, 90.0)
                lon_range = (0.0, 360.0)
            elif (hem == "CONUS"):
                lat_range = (25.0, 50.0) #! changed from 27 to 25
                lon_range = (235.0, 295.0)
            else:
                msg = 'hemispheres must be: GLOBAL, NH, TR, SH, CONUS, or None'
                raise ValueError(msg)

        # Initialize mask
        mask = [True] * len(station_id)

        #filter for the desired station_ids
        if station_ids is not None:
            st_id_mask = [False] * len(station_ids)
            for ids in station_ids: # loop over all station_ids provided
                st_id_mask = np.logical_or(st_id_mask, station_id == id)
            
            mask = np.logical_and(mask, st_id_mask)  

        #filter for the desired obs types
        if obs_types is not None:
            obs_type_mask = [False] * len(obs_type)
            for types in obs_types: # loop over all obs_types provided
                obs_type_mask = np.logical_or(obs_type_mask, obs_type == types)
            
            mask = np.logical_and(mask, obs_type_mask)

        # filter for desired use flag if provided
        if use is not None:
            mask = np.logical_and(mask, use == use_flag)

        # filter by pressure if provided
        if p_range is not None:
            mask = np.logical_and(mask, np.logical_and(pressure >= p_range[0], pressure <= p_range[1]))

        # filter by elevation if provided
        if elv_range is not None:
            mask = np.logical_and(mask,
                                  np.logical_and(elevation >= elv_range[0], elevation <= elv_range[1]))

        # filter bound by lats if provided
        if lat_range is not None:
            mask = np.logical_and(mask, np.logical_and(lat >= lat_range[0], lat <= lat_range[1]))

        # filter bound by lons if provided
        if lon_range is not None:
            mask = np.logical_and(mask, np.logical_and(lon >= lon_range[0], lon <= lon_range[1]))

        #set error mins and maxs, invert them, and filter for desired error values if provided
        if err_range is not None:
            error_min = err_range[0]
            error_max = err_range[1]
            
            mask = np.logical_and(mask, np.logical_and(error >= error_min,
                                                       error <= error_max))

        if(not all(mask)):  
            df.drop(df[~mask].index, inplace = True)
        
        fil_dfs.append(df)

    return fil_dfs