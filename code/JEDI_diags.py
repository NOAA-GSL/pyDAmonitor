import numpy as np
import pandas as pd
import os
from netCDF4 import Dataset

def read_jedi(file_dir, filename, varname, flag=0):

    # Load the NetCDF file
    nc_file = os.path.join(file_dir, filename)
    nc_data = Dataset(nc_file, "r")

    observations = nc_data.groups["ObsValue"].variables[varname][:]
    qc_flags = nc_data.groups["EffectiveQC0"].variables[varname][:]
    
    flag_mask = []
    if flag is not None: # filter by QC
        flag_mask = (qc_flags == flag)
    else: # get all obs
        flag_mask = ~observations.mask
        
    latitude = nc_data.groups["MetaData"].variables["latitude"][:][flag_mask]
    longitude = nc_data.groups["MetaData"].variables["longitude"][:][flag_mask]
    pressure = nc_data.groups["MetaData"].variables["pressure"][:][flag_mask]
    elevation = nc_data.groups["MetaData"].variables["stationElevation"][:][flag_mask]
    station_id = nc_data.groups["MetaData"].variables["stationIdentification"][:][flag_mask]
    observation_type = nc_data.groups["MetaData"].variables["prepbufrReportType"][:][flag_mask]
    #     error = nc_data.groups["DerivedObsError"].variables[varname][:][flag_mask]
    error = nc_data.groups["EffectiveError0"].variables[varname][:][flag_mask]
    #     time = nc_data.groups["MetaData"].variables["dateTime"][:][flag_mask]
    oman = nc_data.groups["oman"].variables[varname][:][flag_mask]
    ombg = nc_data.groups["ombg"].variables[varname][:][flag_mask]
    observations = observations[:][flag_mask]
    qc_flags = qc_flags[:][flag_mask]
    analysis_use_flag = (qc_flags == 0)

    data_jedi = np.column_stack((latitude, longitude, pressure, elevation,
                                 station_id, observation_type, error, oman, ombg, 
                                 observations, qc_flags, analysis_use_flag))
    df_jedi = pd.DataFrame(data_jedi, columns=['latitude', 'longitude', 'pressure', 'elevation',
                                               'station_id', 'observation_type', 'error', 'oman',
                                              'ombg', 'observation', 'qc_flags', 'analysis_use_flag'])
    nc_data.close()

    return df_jedi