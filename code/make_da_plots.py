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

def make_base_plots(df_anl, df_ges, metadata, save_plots=False, **args):

    '''
    df_anl  : pd dataframe with analysis returned by pyDAmonitor.diags.get_data() 
    df_ges  : pd dataframe with guess/forecast returned by pyDAmonitor.diags.get_data() 
    metadata: metadata dict returned by Conventional.metadata from either ges or anl file
    '''
    
    # print(len(args))
    
    # check if the dfs passed are contain wind, which would have speed and direction
    if(metadata['Variable'] == 'uv'):
        print("Error: Use wind_base_plots for wind datasets, not base_plots")
        return None
    
    # Variable name dict
    units_mapping = {
        't': ['Temperature', 'Degrees Fahrenheit'],
        'ps': ['Surface Pressure', 'Pascals'],
        'q': ['Specfic Humidity', 'G Water Vapor per KG of Air']
    }

    # Initialize var_units based on metadata
    var_units = units_mapping.get(metadata['Variable'], None)[1]
    var_name = units_mapping.get(metadata['Variable'], None)[0]
    
    date = metadata['Date'].strftime('%Y%m%d%H')

    # Perform conversions from K to F if the variable is temperature
    if metadata['Variable'] == 't':
        df_anl['observation'] = (1.8 * (df_anl['observation'] - 273.15)) + 32
        df_anl['omf_adjusted'] = 1.8 * df_anl['omf_adjusted']
        df_ges['observation'] = (1.8 * (df_ges['observation'] - 273.15)) + 32
        df_ges['omf_adjusted'] = 1.8 * df_ges['omf_adjusted']
        
    #For saving plots if specified
    dir_name = None
    if(save_plots):
        dir_name = metadata['Variable'] + "_" + date + "_plots"
        os.makedirs(dir_name, exist_ok=True)
        
    print(f'------------ {var_name} Data Assimilation Statistics and Plots ------------\n\n')
        
    #* Create the bar plot by obs type showing proportional assimilated and total assimilated
    if(len(df_anl['observation_type'].unique()) > 1):
        assimilated_by_obs_plots(df_anl, dir_name=dir_name)
    else:
        prop_assimilated = df_anl['analysis_use_flag'].mean()
        ob_type = df_anl['observation_type'].iloc[0]
        print(f'Observation Type: {ob_type}\n\nProportion Assimilated: {prop_assimilated}\n')
    
    #*Plot histograms of obs, omf, and omb
    obs = df_ges['observation']
    omf = df_ges['omf_adjusted']
    oma = df_anl['omf_adjusted']
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))

    #TODO Add customized labels based on variable type
    datasets = [(obs, 'Obs'),
                (omf, 'OmF'),
                (oma, 'OmA')]

    for ax, (data, label) in zip(axes, datasets):
        # Get basic statistics
        n, mean, std, mx, mn = len(data), np.mean(data), np.std(data), np.max(data), np.min(data)
        print(f'{label} Statistics: \nn: {n}, mean: {mean}, std: {std}, max: {mx}, min: {mn}\n')
        
        # Make proper bin sizes using the equation max-min/sqrt(n). Then
        # extend the bin range to 4x the standard deviation
        binsize = (mx - mn) / np.sqrt(n)
        bins = np.arange(mean - (4 * std), mean + (4 * std), binsize)
        # Plot histogram
        ax.hist(data, bins=bins)
        # Add labels
        ax.set_xlabel(var_units)
        ax.set_ylabel('Count')
        ax.set_title(f'{var_name} {label} Histogram', fontsize=14)

    # Add some more space between subplots
    plt.subplots_adjust(wspace=0.3)
    
    if(save_plots):
        plt.savefig(f'{dir_name}/{var_name}_histograms', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    #* Create spatial plots of obs, omf, and oma
    #get lats and lons
    latlons_ges = (df_ges['latitude'], df_ges['longitude'])
    latlons_anl = (df_anl['latitude'], df_anl['longitude'])
    
    #? what is the best way to handle having small scale base_plots and large scale base_plots?
    
    area_size = (latlons_ges[0].max()-latlons_ges[0].min())*(latlons_ges[1].max()-latlons_ges[1].min())
    
    if(area_size<50): #(50 is arbitrary, meant to signify "zoomed in")
        #create small scale spatial plots 
        
        #create obs spatial plot
        fig, ax = plt.subplots(1, 1, figsize=(12,8),
                               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=260)})
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.STATES, edgecolor='black', linewidth = 0.5)
        ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.10, edgecolor='black')
        # ax.set_extent([-180, 180, -90, 90])
        # Plot the scatter data with smaller and more transparent points
        cs = ax.scatter(latlons_ges[1], latlons_ges[0], c=obs, s=20, cmap='Reds', alpha=0.7,
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
            plt.savefig(f'{dir_name}/{var_name}_obs_map')
            plt.close(fig)
        else:
            plt.show()
        
        #create spatial plot with omf and oma
        fig, ax = plt.subplots(1, 1, figsize=(12,8),
                               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=260)})
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.STATES, edgecolor='black')
        ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.10, edgecolor='black')
        # ax.set_extent([-180, 180, -90, 90])
        # Normalization for colorbar
        norm = mcolors.TwoSlopeNorm(vmin = 0 - (2*np.std(omf)), vcenter=0, vmax = 0 + np.std(omf))
        # # Plot the omf/oma data with two circle of different sizes (Option 1)
#         cs1 = ax.scatter(latlons_ges[1], latlons_ges[0], c=omf, s=120, cmap='PRGn',
#                          transform=ccrs.PlateCarree(),
#                          edgecolors='black', linewidths=0.2, label='OmF', norm=norm)
        
#         cs2 = ax.scatter(latlons_anl[1], latlons_anl[0], c=oma, s=35, cmap='PRGn',
#                          transform=ccrs.PlateCarree(),
#                          edgecolors='black', linewidths=0.2, label='OmA', norm=norm)
        # Plot the omf/oma data with two sqaures next to each other (Option 2)
        lat_range = latlons_ges[0].max() - latlons_ges[0].min()
        offset_ratio = 145
        offset = lat_range / offset_ratio
        cs1 = ax.scatter(latlons_anl[1], latlons_anl[0]+offset, c=oma, s=30, cmap='PRGn', marker='s', 
                         edgecolors='black', linewidths=0.7, transform=ccrs.PlateCarree(),
                         label='Top - OmA', norm=norm)
        cs2 = ax.scatter(latlons_ges[1], latlons_ges[0]-offset, c=omf, s=30, cmap='PRGn', marker='s', 
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
            plt.savefig(f'{dir_name}/omf_oma_map')
            plt.close(fig)
        else:
            plt.show()
        
    else:
        # Create large scale spatial subplots
        fig, axes = plt.subplots(3, 1, figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=260)})
        
        datasets = [(obs, 'Obs', 'Reds', latlons_ges),
                    (omf, 'OmF', 'PRGn', latlons_ges),
                    (oma, 'OmA', 'PRGn', latlons_anl)]
        
        #? Add extra dataset here to data_list if there are more than 2?

        #Plot each of the datasets spatially, obs, oma, and omf
        for ax, (data, label, color, coords) in zip(axes, datasets):
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
            ax.add_feature(cfeature.STATES, edgecolor='black')
            # ax.set_extent([-180, 180, -90, 90])
            # Normalization for diverging cmaps, none for increasing cmaps for obs map
            if(label == "Obs"):
                norm=None
            else:
                norm = mcolors.TwoSlopeNorm(vmin = 0 - (np.std(data)*2), vcenter=0, vmax = 0 + (np.std(data)*2))
            # Plot the scatter data with smaller and more transparent points
            cs = ax.scatter(coords[1], coords[0], c=data, s=20, cmap=color, alpha=0.7,
                            transform=ccrs.PlateCarree(), norm=norm)
            # Add gridlines for latitude and longitude
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0)
            gl.top_labels = False
            gl.right_labels = False
            # Add a colorbar
            cb = plt.colorbar(cs, ax=ax, shrink=0.2, pad=.04, extend='both')
            cb.set_label(var_units)
            # Set title
            ax.set_title(f'{var_name} {label} Map', fontsize=14)
            
        plt.tight_layout()
        
        if(save_plots):
            plt.savefig(f'{dir_name}/{var_name}_obs_omf_oma_maps')
            plt.close(fig)
        else:
            plt.show()
        
        
def make_wind_base_plots(df_anl, df_ges, metadata, save_plots=False, **args):
    
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
    
    print('------------ Wind Data Assimilation Statistics and Plots ------------\n\n')
    
    # Get data arrays
    u_obs = df_ges['u_observation']
    v_obs = df_ges['v_observation']
    u_omf = df_ges['u_omf_adjusted']
    v_omf = df_ges['v_omf_adjusted']
    u_oma = df_anl['u_omf_adjusted']
    v_oma = df_anl['v_omf_adjusted']
    
    #* Create the bar plot by obs type showing proportional assimilated and total assimilated
    if(len(df_anl['observation_type'].unique()) > 1):
        assimilated_by_obs_plots(df_anl)
    else:
        prop_assimilated = df_anl['analysis_use_flag'].mean()
        ob_type = df_anl['observation_type'].iloc[0]
        print(f'Observation Type: {ob_type}\n\nProportion Assimilated: {prop_assimilated}\n')
    
    #*Plot histograms of obs, omf, and omb
    #get windspped or wind magnitude from two vectors
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

    axes = axes.flatten()
    for ax, (data, label) in zip(axes, datasets):
        # Get basic statistics
        n, mean, std, mx, mn = len(data), np.mean(data), np.std(data), np.max(data), np.min(data)
        print(f'{label} Statistics: \nn: {n}, mean: {mean}, std: {std}, max: {mx}, min: {mn}\n')
        
        # Make proper bin sizes using the equation max-min/sqrt(n). Then
        # extend the bin range to 4x the standard deviation
        binsize = (mx - mn) / np.sqrt(n)
        bins = np.arange(mean - (4 * std), mean + (4 * std), binsize)
        # Plot histogram
        ax.hist(data, bins=bins)
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
    
    
    # #* Create spatial plots of obs, omf, and oma
    
    # Define your datasets and metadata
    datasets = [
        ('Obs', df_ges, u_obs, v_obs, mag_obs),
        ('OmF', df_ges, u_omf, v_omf, mag_omf),
        ('OmA', df_anl, u_oma, v_oma, mag_oma)
    ]
    
    #Loop thru each dataset
    for title, df, u, v, mag in datasets:
        latlons = (df['latitude'], df['longitude'])
        
        # Get size of geographic extent
        area_size = (latlons[0].max() - latlons[0].min()) * (latlons[1].max() - latlons[1].min())
        
        fig = plt.figure(figsize=(15, 10))
        
        # Add maps features
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=260))
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.STATES, edgecolor='black')
        if area_size < 50:
            ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.10, edgecolor='black')
        
        # Calculate dynamic scale based on the plot dimensions and reference arrow length
        x_range = latlons[1].max() - latlons[1].min()
        y_range = latlons[0].max() - latlons[0].min()
        
        # Normalize magnitude so the arrows are all the same size
        u_norm = u / np.abs(mag)
        v_norm = v / np.abs(mag)
        
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
            plt.savefig(f'{dir_name}/wind_speed_direction_{title}_map')
            plt.close(fig)
        else:
            plt.show()
    
    
def assimilated_by_obs_plots(df_anl, dir_name):
    
    '''
    Create the bar plot by obs type showing proportional assimilated and total assimilated
    
    If dir_name is None, then we know save_plots is False
    '''
    
    #calculated prop and totals
    prop_assim_by_obs_type_anl = pd.DataFrame(df_anl.groupby('observation_type')['analysis_use_flag'].mean())
    total_assim_by_obs_type_anl = pd.DataFrame(df_anl.groupby('observation_type')['analysis_use_flag'].sum())
    
    fig, axes = plt.subplots(1, 2, figsize = (18, 6))
    
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
        plt.savefig(f'{dir_name}/{var_name}_{label}_map')
        plt.clf('assimilated_sum_by_obstype')
        