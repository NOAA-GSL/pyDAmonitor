'''
File taken from PyGSI
Altercations make for PyGSI:

-Made all df column headers all lowercase
-Removed code that creates the multilayer indices in the df
-Changed analysis use flags that were -1 to be 0
-Removed Ozone and Radiance classes for now
'''


import os
import io
import sys
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from pathlib import Path
import gzip

_VALID_LVL_TYPES = ["pressure", "height"]


class GSIdiag:

    def __init__(self, path):
        """
        Initialize a GSI diagnostic object
        INPUT:
            path : path to GSI diagnostic object
        """

        self.path = path
        self.filename = os.path.splitext(Path(self.path).stem)[0]
        self.obs_type = self.filename.split('_')[1]
        self.variable = self.filename.split('_')[2]
        self.ftype = self.filename.split('_')[-1]

        _str_date = os.path.basename(self.path).split('.')[1]
        # Checks if '_ensmean' is included in file name
        self.date = datetime.strptime(_str_date.split('_')[0], '%Y%m%d%H')

        _var_key_name = 'Variable' if self.obs_type == 'conv' else 'Satellite'
        self.metadata = {'Obs Type': self.obs_type,
                         _var_key_name: self.variable,
                         'Date': self.date,
                         'File Type': self.ftype
                         }

    # def __len__(self):
    #     return self.data_df.shape[0]


class Conventional(GSIdiag):

    def __init__(self, path):
        """
        Initialize a conventional GSI diagnostic object

        Args:
            path : (str) path to conventional GSI diagnostic object
        Returns:
            self : GSI diag conventional object containing the path
                   to extract data
        """
        super().__init__(path)

        self._read_obs()
        self.metadata['Diag File Type'] = 'conventional'
        
    def __len__(self):
        return self.data_df.shape[0]

    def __str__(self):
        return "Conventional GSI diagnostic object"

    def _read_obs(self):
        """
        Reads the data from the conventional diagnostic file during
        initialization into a multidimensional pandas dataframe.
        """
        
        df_dict = {}
        f = None
        
        try:
            #Added by Aiden, capability to read gzipped netCDF files
            if(self.path.endswith('.gz')):
                with gzip.open(self.path, 'rb') as f_gz:
                   # Read the file into a buffer (in memory not disk)
                   buffer = io.BytesIO(f_gz.read())
                   # Open from the buffer
                   f = Dataset('dummy', mode='r', memory=buffer.read())
            else:
                f = Dataset(self.path, mode='r')

            # Open netCDF file, store data into dictionary
            for var in f.variables:
                # station_id and observation_class variables need
                # to be converted from byte string to string
                if var in ['Station_ID', 'Observation_Class']:
                    data = f.variables[var][:]
                    data = [i.tobytes(fill_value='/////', order='C')
                            for i in data]
                    data = np.array(
                        [''.join(i.decode('UTF-8', 'ignore').split())
                         for i in data])
                    df_dict[var] = data

                # Grab variables with only 'nobs' dimension
                elif len(f.variables[var].shape) == 1:
                    df_dict[var] = f.variables[var][:]
                    
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(-1)
        
        # Ensure we close the netcdf file since we didn't use 'with'
        finally:
            if f is not None:
                f.close()

        # Create pandas dataframe from dict
        df = pd.DataFrame(df_dict)

        #! Removed by Aiden for pyDAMonitor
        # Creates multidimensional indexed dataframe
        # indices = ['station_id', 'observation_class', 'observation_type',
        #            'observation_subtype', 'pressure', 'height',
        #            'analysis_use_flag']
        # df.set_index(indices, inplace=True)

        # Rename columns
        df.columns = df.columns.str.lower()
        if self.variable == 'uv':
            for wind_type in ['u', 'v']:
                for bias_type in ['unadjusted', 'adjusted']:
                    df = df.rename(columns={
                        f'{wind_type}_obs_minus_forecast_{bias_type}':
                        f'{wind_type}_omf_{bias_type}'
                        })
                    # Create hofx columns
                    df[f'{wind_type}_hofx_{bias_type}'] = \
                        df[f'{wind_type}_observation'] - \
                        df[f'{wind_type}_omf_{bias_type}']

        else:
            for bias_type in ['unadjusted', 'adjusted']:
                df = df.rename(columns={
                    f'obs_minus_forecast_{bias_type}': f'omf_{bias_type}',
                    })
                # Create hofx columns
                df[f'hofx_{bias_type}'] = df['observation'] - \
                    df[f'omf_{bias_type}']

        #! Changed by Aiden for pyDAmonitor
        # # Get index values
        # self.obs_ids = df.index.get_level_values(
        #     'observation_type').unique().to_numpy()
        # self.stn_ids = df.index.get_level_values(
        #     'station_id').unique().to_numpy()
        self.obs_ids = df['observation_type'].unique()
        self.stn_ids = df['station_id'].unique()
        
        #* Added by Aiden for pyDAmonitor
        df.loc[df['analysis_use_flag'] == -1, 'analysis_use_flag'] = 0

        self.data_df = df

    def get_data(self, obsid=None, subtype=None, station_id=None,
                 analysis_use=False, lvls=None, lvl_type='pressure'):
        """
        Given parameters, get indexed dataframe from a conventional diagnostic
        file.

        Args:
            obsid : (list of ints; optional; default=None) observation
                    type ID number; default=None
            subtype : (list of ints; optional; default=None) observation
                      measurement ID subtype number, default=None
            station_id : (list of str; optional; default=None)
                         station id, default=None
            analysis_use : (bool; default=False) if True, will return
                           three sets of data:
                           assimilated (analysis_use_flag=1, qc<7),
                           rejected (analysis_use_flag=0, qc<8), #! Changed by Aiden
                           monitored (analysis_use_flag=0, qc>7) #! Changed by Aiden
            lvls : (list type; default=None) List of pressure or height
                   levels i.e. [250,500,750,1000]. List must be arranged
                   low to high.
                   For pressure, will return a dictionary of data with data
                   greater than the low bound, and less than or equal to the
                   high bound.
                   For height, will return a dictionary of data with data
                   greater than or equal to the low bound, and less than the
                   high bound.
            lvl_type : (str; default='pressure') lvls definition as
                       'pressure' or 'height'.
        Returns:
            indexed_df : requested indexed dataframe
        """

        self.metadata['ObsID'] = obsid
        self.metadata['Subtype'] = subtype
        self.metadata['Station ID'] = station_id
        self.metadata['Anl Use'] = analysis_use
        self.metadata['Levels'] = lvls
        self.metadata['Levels Type'] = lvl_type

        # Selects proper levels
        if lvls is not None:
            # Check if level type is valid
            if lvl_type not in _VALID_LVL_TYPES:
                raise ValueError((f'{lvl_type} is not a valid lvl_type. '
                                  'Valid choices are: '
                                  f'{" | ".join(_VALID_LVL_TYPES)}'))

            binned_dfs = self._get_lvl_data(obsid, subtype, station_id,
                                            analysis_use, lvls, lvl_type)

            return binned_dfs

        else:
            if analysis_use:
                assimilated_df, rejected_df, monitored_df = self._select_conv(
                    obsid, subtype, station_id, analysis_use)

                indexed_df = {
                    'assimilated': assimilated_df,
                    'rejected': rejected_df,
                    'monitored': monitored_df
                }

            else:
                indexed_df = self._select_conv(obsid, subtype, station_id,
                                               analysis_use)

            return indexed_df

    def _select_conv(self, obsid, subtype, station_id, analysis_use):
        """
        Given parameters, multidimensional dataframe is indexed
        to only include selected locations from a conventional
        diagnostic file.

        Returns:
            df : (pandas dataframe) indexed multidimentsional
                 dataframe from selected data
        """

        df = self.data_df

        # select data by obsid, subtype, and station ids
        if obsid is not None:
            idx_col = 'observation_type'
            indx = df.index.get_level_values(idx_col) == ''
            for obid in obsid:
                indx = np.ma.logical_or(
                    indx, df.index.get_level_values(idx_col) == obid)
            df = df.iloc[indx]
        if subtype is not None:
            idx_col = 'observation_subtype'
            indx = df.index.get_level_values(idx_col) == ''
            for stype in subtype:
                indx = np.ma.logical_or(
                    indx, df.index.get_level_values(idx_col) == stype)
            df = df.iloc[indx]
        if station_id is not None:
            idx_col = 'station_id'
            indx = df.index.get_level_values(idx_col) == ''
            for stn_id in station_id:
                indx = np.ma.logical_or(
                    indx, df.index.get_level_values(idx_col) == stn_id)
            df = df.iloc[indx]

        if analysis_use:
            # Separate into 3 dataframes; assimilated, rejected, and monitored
            indx = df.index.get_level_values('analysis_use_flag') == ''

            assimilated_indx = np.ma.logical_or(
                indx, df.index.get_level_values('analysis_use_flag') == 1)
            rejected_indx = np.ma.logical_or(
                indx, df.index.get_level_values('analysis_use_flag') == -1)
            monitored_indx = np.ma.logical_or(
                indx, df.index.get_level_values('analysis_use_flag') == -1)

            assimilated_df = df.iloc[assimilated_indx]
            rejected_df = df.iloc[rejected_indx]
            monitored_df = df.iloc[monitored_indx]

            # Find rejected and monitored based on Prep_QC_Mark
            try:
                assimilated_df = assimilated_df.loc[
                    assimilated_df['prep_qc_mark'] < 7]
                rejected_df = rejected_df.loc[
                    rejected_df['prep_qc_mark'] < 8]
                monitored_df = monitored_df.loc[
                    monitored_df['prep_qc_mark'] > 7]
            except KeyError:
                assimilated_df = assimilated_df.loc[
                    assimilated_df['setup_qc_mark'] < 7]
                rejected_df = rejected_df.loc[
                    rejected_df['setup_qc_mark'] < 8]
                monitored_df = monitored_df.loc[
                    monitored_df['setup_qc_mark'] > 7]

            return assimilated_df, rejected_df, monitored_df

        else:
            return df

    def _get_lvl_data(self, obsid, subtype, station_id,
                      analysis_use, lvls, lvl_type):
        """
        Given a list of levels, will create a dictionary of data that is
        selected between each level. Will return a dictionary with subsetted
        pressure or height levels where data is separated within those levels:

        dict = {250-500: <data>,
                500-750: <data>,
                750-1000: <data>}
        """
        binned_dfs = {}

        for i, low_bound in enumerate(lvls[:-1]):
            high_bound = lvls[i+1]

            if analysis_use:
                assimilated_df, rejected_df, monitored_df = self._select_conv(
                    obsid, subtype, station_id, analysis_use)

                assimilated_lvl_df = self._select_levels(
                    assimilated_df, low_bound, high_bound, lvl_type)
                rejected_lvl_df = self._select_levels(
                    rejected_df, low_bound, high_bound, lvl_type)
                monitored_lvl_df = self._select_levels(
                    monitored_df, low_bound, high_bound, lvl_type)

                lvl_df = {
                    'assimilated': assimilated_lvl_df,
                    'rejected': rejected_lvl_df,
                    'monitored': monitored_lvl_df
                }

                binned_dfs[f'{low_bound}-{high_bound}'] = lvl

            else:
                indexed_df = self._select_conv(obsid, subtype, station_id)

                lvl_df = self._select_levels(
                    indexed_df, low_bound, high_bound, lvl_type)

                binned_dfs[f'{low_bound}-{high_bound}'] = lvl_df

        return binned_dfs

    def _select_levels(self, df, low_bound, high_bound, lvl_type):
        """
        Selects data between two level bounds from given dataframe.
        """

        if lvl_type == 'pressure':
            # Grab data greater than low bound and less than or
            # equal to high_bound
            df = df.query(
                f'(pressure > {low_bound}) and (pressure <= {high_bound})')

        else:
            # Grab data greater than or equal to low bound and
            # less than high_bound
            df = df.query(
                f'(height >= {low_bound}) and (height < {high_bound})')

        return df

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

