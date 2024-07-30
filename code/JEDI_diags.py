import numpy as np
import pandas as pd
import os
from netCDF4 import Dataset
from pathlib import Path


class JEDIdiag:

    def __init__(self, path, varname):
        """
        Initialize a GSI diagnostic object
        INPUT:
            path : path to GSI diagnostic object
        """

        self.path = path
        self.filename = os.path.splitext(Path(self.path).stem)[0]

        self.metadata = {'Variable': varname,
                         'Date': "Not implemented yet",
                         'File Type': "JEDI H(x) diag file"
                         }

class hofx(JEDIdiag):

    def __init__(self, path, varname, flag=0):
        """
        Initialize a conventional GSI diagnostic object

        Args:
            path : (str) path to conventional GSI diagnostic object
        Returns:
            self : GSI diag conventional object containing the path
                   to extract data
        """
        super().__init__(path, varname)

        self.df = self.read_obs(varname, flag)
        
    def __len__(self):
        return self.data_df.shape[0]

    def __str__(self):
        return "JEDI H(x) diagnostic object"

    def read_obs(self, varname, flag):

        # Load the NetCDF file
        nc_data = Dataset(self.path, "r")
        
        var_mapping = {
            't': 'airTemperature',
            'ps': 'surfacePressure',
            'q': 'specificHumidity',
        }
        
        varname = var_mapping.get(varname)

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
        station_elevation = nc_data.groups["MetaData"].variables["stationElevation"][:][flag_mask]
        station_id = nc_data.groups["MetaData"].variables["stationIdentification"][:][flag_mask]
        observation_type = nc_data.groups["MetaData"].variables["prepbufrReportType"][:][flag_mask]
        #     error = nc_data.groups["DerivedObsError"].variables[varname][:][flag_mask]
        error = nc_data.groups["EffectiveError0"].variables[varname][:][flag_mask]
        #     time = nc_data.groups["MetaData"].variables["dateTime"][:][flag_mask]
        oman = nc_data.groups["oman"].variables[varname][:][flag_mask]
        ombg = nc_data.groups["ombg"].variables[varname][:][flag_mask]
        datetime = nc_data.groups["MetaData"].variables["dateTime"][:][flag_mask]
        observations = observations[:][flag_mask]
        qc_flags = qc_flags[:][flag_mask]
        analysis_use_flag = (qc_flags == 0)

        data_jedi = np.column_stack((datetime, latitude, longitude, pressure, station_elevation,
                                     station_id, observation_type, error, oman, ombg, 
                                     observations, qc_flags, analysis_use_flag))
        df_jedi = pd.DataFrame(data_jedi, columns=['datetime', 'latitude', 'longitude', 'pressure',
                                                   'station_elevation', 'station_id', 'observation_type',
                                                   'error', 'oman', 'ombg', 'observation', 'qc_flags',
                                                   'analysis_use_flag'])
        nc_data.close()

        return df_jedi
    


    def list_obsids(self):
        """
        Prints all the unique observation ids in the diagnostic file.
        """
        obsids = sorted(self.obs_ids)

        print('Observation IDs in this diagnostic file include:\n'
              f'{", ".join([str(x) for x in obsids])}')

    def list_stationids(self):
        """
        Prints all the unique station ids in the diagnostic file.
        """
        stnids = sorted(self.stn_ids)

        print('Station IDs in this diagnostic file include:\n'
              f'{", ".join([str(x) for x in stnids])}')


