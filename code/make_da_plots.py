''' 
Author: Aiden Pape
Date: Jun 13, 2024

Automate basic plot display
'''

import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES

def make_base_plots(dfs, metadata, zoom=True, save_plots=False, shared_norm = None, hem=None, **args):

    '''
    dfs        : list of dataframes to make plots of, should be ordered as followes (ges, anl, anl2, etc)
                 all dataframes should be returned by pyDAmonitor.diags.get_data() (at the moment, only 
                 [ges, anl] supported
    metadata   : metadata dict returned by Conventional.metadata from either ges or anl file
    zoom       : (bool) True to plot oma/omf on the same plot on more zoomed in geo extents
    save_plots : (bool) True to save plots as files instead of printing
    shared_norm: omf/oma norm returned from other make_base_plots function if aligned scales is needed
    
    Additional Args:
    All keyword arguments to filter can be passed to make_base_plots to filter data before creating plots
    
    return: to_share_norm, the norm used in these omf/oma plots to be passed to another make_base_plots 
    '''
    
    # check if the dfs passed are contain wind, which would have speed and direction
    if(metadata['Variable'] == 'uv'):
        print("Error: Use wind_base_plots for wind datasets, not base_plots")
        return None
    
    #Norm to pass to our make_base_plots to have consistent scaling
    to_share_norm = None
    
    # Variable name dict
    units_mapping = {
        't': ['Temperature', 'Degrees Fahrenheit'],
        'ps': ['Surface Pressure', 'Pascals'],
        'q': ['Specific Humidity', 'G Water Vapor per KG of Air']
    }

    # Initialize var_units based on metadata
    var_units = units_mapping.get(metadata['Variable'])[1]
    var_name = units_mapping.get(metadata['Variable'])[0]
    
    date = metadata['Date'].strftime('%Y%m%d%H')
        
    #For saving plots if specified
    dir_name = None
    if(save_plots):
        dir_name = "plots/" + metadata['Variable'] + "_" + date + "_plots"
        os.makedirs(dir_name, exist_ok=True)
        
    lat_range = None
    lon_range = None
#     if hem is provided
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
    
    #Get data from dfs based on which system, (GSI has two files for ges and anl, JEDI has all in one)
    df_ges=None
    df_anl = None
    obs = None
    omf = None
    oma = None
    latlons_ges = None
    latlons_anl = None
    
    if(metadata['DA System'] == 'gsi'):
        
        df_ges = dfs[0]
        df_anl = dfs[1]

        obs = df_ges['observation']
        omf = df_ges['omf_adjusted']
        oma = df_anl['omf_adjusted']
        
        latlons_ges = (df_ges['latitude'], df_ges['longitude'])
        latlons_anl = (df_anl['latitude'], df_anl['longitude'])
        
    elif(metadata['DA System'] == 'jedi'):
        
        df_anl = dfs[0]
        
        obs = df_anl['observation']
        omf = df_anl['ombg']
        oma = df_anl['oman']
        
        # this seemed like the easiest way
        latlons_ges = (df_anl['latitude'], df_anl['longitude'])
        latlons_anl = (df_anl['latitude'], df_anl['longitude'])
        
    if(not save_plots):
        print(f'------------ {var_name} Data Assimilation Statistics and Plots ------------\n\n')
    else:
        print(f'Creating {var_name} plots...')
        
    #* Create the bar plot by obs type showing proportional assimilated and total assimilated
    if(len(df_anl['observation_type'].unique()) > 1):
        assimilated_by_obs_plots(df_anl, dir_name=dir_name)
    else:
        prop_assimilated = df_anl['analysis_use_flag'].mean()
        ob_type = df_anl['observation_type'].iloc[0]
        print(f'Observation Type: {ob_type}\n\nProportion Assimilated: {prop_assimilated}\n')
    
    # Perform conversions from K to F if the variable is temperature
    if metadata['Variable'] == 't':
        obs = (1.8 * (obs - 273.15)) + 32
        omf = 1.8 * omf
        oma = 1.8 * oma
        
    #*Plot histograms of obs, omf, and omb
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))

    #TODO Add customized labels based on variable type
    datasets = [(obs, 'Obs'),
                (omf, 'OmF'),
                (oma, 'OmA')]
    
    statistics = []
    dp = 3 # number of decimal places for stats

    for ax, (data, label) in zip(axes, datasets):
        #convert to float for decimal precision
        data = np.array(data, dtype=float)
        # Get basic statistics
        n, mean, std, mx, mn = ( len(data), 
                                np.round(np.mean(data), dp),
                                np.round(np.std(data), dp),
                                np.round(np.max(data), dp),
                                np.round(np.min(data), dp)
                                )
        statistics.append([label, n, mean, std, mx, mn])
        # print(f'{label} Statistics: \nn: {n}, mean: {mean}, std: {std}, max: {mx}, min: {mn}\n')
        
        # Make proper bin sizes using the equation max-min/sqrt(n). Then
        # extend the bin range to 4x the standard deviation
#         binsize = (mx - mn) / np.sqrt(n)
#         bins = np.arange(mean - (3 * std), mean + (3 * std), binsize)
        # Plot histogram
        ax.hist(data, bins='auto')
        # Add labels
        ax.set_xlabel(var_units)
        ax.set_ylabel('Count')
        ax.set_title(f'{var_name} {label} Histogram', fontsize=14)

    # Add some more space between subplots
    plt.subplots_adjust(wspace=0.3)
    
    # save or show plots
    if(save_plots):
        plt.savefig(f'{dir_name}/{var_name}_histograms', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
    # Create DataFrame for statistics
    stats_df = pd.DataFrame(statistics, columns=['Label', 'Count', 'Mean', 'Std Dev', 'Max', 'Min'])
    
    # Create a figure for statistics figure
    fig, ax = plt.subplots(figsize = (12,2))

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    ax.table(cellText=stats_df.values, colLabels=stats_df.columns, cellLoc='center', loc='center')

    # save or show plots
    if(save_plots):
        plt.savefig(f'{dir_name}/{var_name}_statistics', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    #* Create spatial plots of obs, omf, and oma
    
    #? what is the best way to handle having small scale base_plots and large scale base_plots?
    
    area_size = (latlons_ges[0].max()-latlons_ges[0].min())*(latlons_ges[1].max()-latlons_ges[1].min())
    
    if area_size<50 and zoom: #(50 is arbitrary, meant to signify "zoomed in")
        #create small scale spatial plots 
        
        #create obs spatial plot
        fig, ax = plt.subplots(1, 1, figsize=(18,16),
                               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=260)})
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.STATES, edgecolor='black', linewidth = 0.5)
        ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.10, edgecolor='black')
        # ax.set_extent([-180, 180, -90, 90])
        # Plot the scatter data with smaller and more transparent points
        cs = ax.scatter(latlons_ges[1], latlons_ges[0], c=obs, s=30, cmap='Reds', alpha=0.7,
                        transform=ccrs.PlateCarree())
        # Add gridlines for latitude and longitude
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        # Add a colorbar
        cb = plt.colorbar(cs, ax=ax, shrink=0.2, pad=.04, extend='both')
        cb.set_label(var_units)
        # Set title
        ax.set_title(f'{var_name} Obs', fontsize=14)
        plt.tight_layout()
        if(save_plots):
            plt.savefig(f'{dir_name}/{var_name}_obs_map', bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        
        #create spatial plot with omf and oma
        fig, ax = plt.subplots(1, 1, figsize=(18,16),
                               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=260)})
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.STATES, edgecolor='black')
        ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.10, edgecolor='black')
        # ax.set_extent([-180, 180, -90, 90])
        # Normalization for colorbar
        if shared_norm is not None:
            norm = shared_norm
        else:
            norm = mcolors.TwoSlopeNorm(vmin = 0 - (np.std(omf)*2), vcenter=0, vmax = 0 + (np.std(omf)*2))
            to_share_norm = norm
        # # Plot the omf/oma data with two circle of different sizes (Option 1)
#         cs1 = ax.scatter(latlons_ges[1], latlons_ges[0], c=omf, s=200, cmap='PRGn',
#                          transform=ccrs.PlateCarree(),
#                          edgecolors='black', linewidths=0.2, label='OmF', norm=norm)
        
#         cs2 = ax.scatter(latlons_anl[1], latlons_anl[0], c=oma, s=50, cmap='PRGn',
#                          transform=ccrs.PlateCarree(),
#                          edgecolors='black', linewidths=0.2, label='OmA', norm=norm)
        # Plot the omf/oma data with two sqaures next to each other (Option 2)
        lat_range = latlons_ges[0].max() - latlons_ges[0].min()
        offset_ratio = 175
        offset = lat_range / offset_ratio
        cs1 = ax.scatter(latlons_anl[1], latlons_anl[0]+offset, c=oma, s=50, cmap='PRGn', marker='s', 
                         edgecolors='black', linewidths=0.7, transform=ccrs.PlateCarree(),
                         label='Top - OmA', norm=norm)
        cs2 = ax.scatter(latlons_ges[1], latlons_ges[0]-offset, c=omf, s=50, cmap='PRGn', marker='s', 
                         edgecolors='black', linewidths=0.7, transform=ccrs.PlateCarree(),
                         label='Bottom - OmF', norm=norm)
        # Plot (OmA - OmF) FmA (Option 3)
#         fma = oma - omf
#         cs1 = ax.scatter(latlons_ges[1], latlons_ges[0], c=fma, s=(np.abs(fma))*15, cmap='PRGn', 
#                          transform=ccrs.PlateCarree(),
#                          edgecolors='black', linewidths=0.2, label='FmA', norm=norm)
        # Add gridlines for latitude and longitude
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        # Add a colorbar adn legend
        cb = plt.colorbar(cs1, ax=ax, shrink=0.2, pad=.04, extend='both')
        cb.set_label(var_units)
        ax.legend(loc='upper right', fontsize='medium', framealpha=0.9)
        # Set title
        ax.set_title(f'{var_name} OmF and OmA Comparison', fontsize=14)
        plt.tight_layout()
        
        if(save_plots):
            plt.savefig(f'{dir_name}/omf_oma_map', bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        
    else:
        # Create large scale spatial subplots
        fig, axes = plt.subplots(3, 1, figsize=(18,16), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=260)})
        
        datasets = [(obs, 'Obs', 'Reds', latlons_ges),
                    (omf, 'OmF', 'PRGn', latlons_ges),
                    (oma, 'OmA', 'PRGn', latlons_anl)]
        
        #? Add extra dataset here to data_list if there are more than 2?

        #Plot each of the datasets spatially, obs, oma, and omf
        for ax, (data, label, color, coords) in zip(axes, datasets):
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
            ax.add_feature(cfeature.STATES, edgecolor='black')
            if(area_size < 50):
                ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.10, edgecolor='black')
            if hem is not None:
                ax.set_extent([lon_range[0]-1, lon_range[1]+1, lat_range[0]-1, lat_range[1]+1])
            # Normalization for diverging cmaps, none for increasing cmaps for obs map
            if(label == "Obs"):
                norm=None
            else:
                if shared_norm is not None:
                    norm = shared_norm
                else:
                    norm = mcolors.TwoSlopeNorm(vmin = 0 - (np.std(omf)*2), vcenter=0, vmax = 0 + (np.std(omf)*2))
                    to_share_norm = norm
            # Plot the scatter data
            cs = ax.scatter(coords[1], coords[0], c=data, s=60, cmap=color, alpha=0.9, edgecolors='black',
                            linewidths=0.5, transform=ccrs.PlateCarree(), norm=norm)
            # Add gridlines for latitude and longitude
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0)
            gl.top_labels = False
            gl.right_labels = False
            # Add a colorbar
            cb = plt.colorbar(cs, ax=ax, shrink=0.8, pad=.04, extend='both')
            cb.set_label(var_units)
            # Set title
            ax.set_title(f'{var_name} {label} Map', fontsize=14)
            
        plt.tight_layout()
        
        if(save_plots):
            plt.savefig(f'{dir_name}/{var_name}_obs_omf_oma_maps', bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            
    if(save_plots):
        print(f'Plots created successfully, saved to in {dir_name} folder')
        
    return to_share_norm
        
        
def make_wind_base_plots(dfs, metadata, save_plots=False, **args):
    
    # check if the dfs passed are contain wind, which would have speed and direction
    if(metadata['Variable'] != 'uv'):
        print("Error: Use base_plots for non wind datasets")
        return None
    
    # Get date from metadata
    date = metadata['Date'].strftime('%Y%m%d%H')
    
    #For saving plots if specified
    dir_name = None
    if(save_plots):
        dir_name = f'wind_{date}_plots'
        os.makedirs(dir_name, exist_ok=True)
        
    #Get data from dfs based on which system, (GSI has two files for ges and anl, JEDI has all in one)
    df_ges=None
    df_anl = None
    obs = None
    omf = None
    oma = None
    latlons_ges = None
    latlons_anl = None
    
    if(metadata['DA System'] == 'gsi'):
        
        df_ges = dfs[0]
        df_anl = dfs[1]

        u_obs = df_ges['u_observation']
        v_obs = df_ges['v_observation']
        u_omf = df_ges['u_omf_adjusted']
        v_omf = df_ges['v_omf_adjusted']
        u_oma = df_anl['u_omf_adjusted']
        v_oma = df_anl['v_omf_adjusted']
        
        latlons_ges = (df_ges['latitude'], df_ges['longitude'])
        latlons_anl = (df_anl['latitude'], df_anl['longitude'])
        
    elif(metadata['DA System'] == 'jedi'):
        
        df_anl = dfs[0]
        
        u_obs = df_anl['u_observation']
        v_obs = df_anl['v_observation']
        u_omf = df_anl['u_ombg']
        v_omf = df_anl['v_ombg']
        u_oma = df_anl['u_oman']
        v_oma = df_anl['v_oman']
        
        # this seemed like the easiest way
        latlons_ges = (df_anl['latitude'], df_anl['longitude'])
        latlons_anl = (df_anl['latitude'], df_anl['longitude'])
    
    if(not save_plots):
        print('------------ Wind Data Assimilation Statistics and Plots ------------\n\n')
    else:
            print('Creating wind plots...')
    
    #* Create the bar plot by obs type showing proportional assimilated and total assimilated
    if(len(df_anl['observation_type'].unique()) > 1):
        assimilated_by_obs_plots(df_anl, dir_name)
    else:
        prop_assimilated = df_anl['analysis_use_flag'].mean()
        ob_type = df_anl['observation_type'].iloc[0]
        print(f'Observation Type: {ob_type}\n\nProportion Assimilated: {prop_assimilated}\n')
    
    #*Plot histograms of obs, omf, and omb
    #get windspeed or wind magnitude from two vectors
    mag_obs = np.sqrt( (u_obs**2) + (v_obs**2) )
    mag_omf = np.sqrt( (u_omf**2) + (v_omf**2) )
    mag_oma = np.sqrt( (u_oma**2) + (v_oma**2) )
        
    mag_omf = np.where((u_omf + v_omf) < 0, -mag_omf, mag_omf)
    mag_oma = np.where((u_oma + v_oma) < 0, -mag_oma, mag_oma)  
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(16, 5))

    #TODO Add customized labels based on variable type
    datasets = [(u_obs, 'U Obs'),
                 (u_omf, 'U OmF'),
                 (u_oma, 'U OmA'),
                 (v_obs, 'V Obs'),
                 (v_omf, 'V OmF'),
                 (v_oma, 'V OmA'),
                 (mag_obs, 'Wind Speed Obs'),
                 (mag_omf, 'Wind Speed OmF'),
                 (mag_oma, 'Wind Speed OmA')]
    
    statistics = []
    dp = 3 # number of decimal places for stats

    axes = axes.flatten()
    for ax, (data, label) in zip(axes, datasets):
        #convert to float for decimal precision
        data = np.array(data, dtype=float)
        # Get basic statistics
        n, mean, std, mx, mn = ( len(data), 
                                np.round(np.mean(data), dp), 
                                np.round(np.std(data), dp),
                                np.round(np.max(data), dp), 
                                np.round(np.min(data), dp) 
                                )
        statistics.append([label, n, mean, std, mx, mn])
        # print(f'{label} Statistics: \nn: {n}, mean: {mean}, std: {std}, max: {mx}, min: {mn}\n')
        
        # Make proper bin sizes using the equation max-min/sqrt(n). Then
        # extend the bin range to 4x the standard deviation
#         binsize = (mx - mn) / np.sqrt(n)
#         bins = np.arange(mean - (4 * std), mean + (4 * std), binsize)
        # Plot histogram
        ax.hist(data, bins='auto')
        # Add labels
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(f'Wind {label} Histogram', fontsize=14)

    # Add some more space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=1)
    
    if(save_plots):
        plt.savefig(f'{dir_name}/wind_histograms', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
    # Create DataFrame for statistics
    stats_df = pd.DataFrame(statistics, columns=['Label', 'Count', 'Mean', 'Std Dev', 'Max', 'Min'])
    
    # Create a figure for statistics figure
    fig, ax = plt.subplots(figsize = (10,2))

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    ax.table(cellText=stats_df.values, colLabels=stats_df.columns, cellLoc='center', loc='center')

    # save or show plots
    if(save_plots):
        plt.savefig(f'{dir_name}/wind_statistics', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    
    # #* Create spatial plots of obs, omf, and oma
    
    # Define your datasets and metadata
    datasets = [
        ('Obs', latlons_ges, u_obs, v_obs, mag_obs),
        ('OmF', latlons_ges, u_omf, v_omf, mag_omf),
        ('OmA', latlons_anl, u_oma, v_oma, mag_oma)
    ]
    
    #Loop thru each dataset
    for title, latlons, u, v, mag in datasets:
        
        lats = latlons[0]
        lons = latlons[1]
        
        # Get size of geographic extent
        area_size = (latlons[0].max() - latlons[0].min()) * (latlons[1].max() - latlons[1].min())
        
        fig = plt.figure(figsize=(18, 10))
        
        # Add maps features
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=260))
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.STATES, edgecolor='black')
        if area_size < 50:
            ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.10, edgecolor='black')
        
        # Normalize magnitude so the arrows are all the same size, init with zeros
        u_norm = np.zeros_like(u)
        v_norm = np.zeros_like(v)

        # Set u_norm and v_norm only where mag is not zero
        non_zero_mask = mag != 0
        u_norm[non_zero_mask] = u[non_zero_mask] / np.abs(mag[non_zero_mask])
        v_norm[non_zero_mask] = v[non_zero_mask] / np.abs(mag[non_zero_mask])
        
        # Plot wind as arrows
        cs = plt.quiver(latlons[1], latlons[0], u_norm, v_norm, mag, scale = 5, scale_units = 'inches',
                        units = 'inches', width = 0.02,
                        cmap='plasma', transform=ccrs.PlateCarree())
        
        # Add a colorbar
        cb = plt.colorbar(cs, shrink=0.2, pad=.04, extend='both')
        cb.set_label('Wind Speed')
        
        # Add gridlines for latitude and longitude
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        plt.title(f"Wind Speed and Direction {title}")
        
        # Display or save the plot
        if(save_plots):
            plt.savefig(f'{dir_name}/wind_speed_direction_{title}_map', bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    if(save_plots):
        print(f'Plots created successfully, saved to in {dir_name} folder')
    
def assimilated_by_obs_plots(df_anl, dir_name):
    
    '''
    Create the bar plot by obs type showing proportional assimilated and total assimilated
    
    If dir_name is None, then we know save_plots is False
    '''
    
    #calculated prop and totals
    prop_assim_by_obs_type_anl = pd.DataFrame(df_anl.groupby('observation_type')['analysis_use_flag'].mean())
    total_assim_by_obs_type_anl = pd.DataFrame(df_anl.groupby('observation_type')['analysis_use_flag'].count())
    
    fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    
    #create bars for prop
    axes[0].bar(range(len(prop_assim_by_obs_type_anl['analysis_use_flag'])),
                prop_assim_by_obs_type_anl['analysis_use_flag'],
                tick_label=prop_assim_by_obs_type_anl.index)
    # Rotate tick labels
    axes[0].tick_params(axis='x', rotation=90)
    # Add labels and title as necessary
    axes[0].set_xlabel('Obs Type')
    axes[0].set_ylabel('Prop. Assimilated')
    axes[0].set_title('Proportion Assimilated Obs by Obs Type')
    #create bars for prop
    axes[1].bar(range(len(total_assim_by_obs_type_anl['analysis_use_flag'])),
                total_assim_by_obs_type_anl['analysis_use_flag'],
                tick_label=prop_assim_by_obs_type_anl.index)
    # Rotate tick labels
    axes[1].tick_params(axis='x', rotation=90)
    # Add labels and title as necessary
    axes[1].set_xlabel('Obs Type')
    axes[1].set_ylabel('Total Assimilated')
    axes[1].set_title('Total Assimilated Obs by Obs Type')
        
    if dir_name is None:
        plt.show()
    else:
        plt.savefig(f'{dir_name}/assimilated_obs_plots', bbox_inches='tight')
        plt.close(fig)
        