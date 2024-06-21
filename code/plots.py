''' 
Author: Aiden Pape
Date: Jun 13, 2024

Automate basic plot display
'''

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def base_plots(df_anl, df_ges, metadata, **args):

    '''
    df_anl  : pd dataframe with analysis returned by pyDAmonitor.diags.get_data() 
    df_ges  : pd dataframe with guess/forecast returned by pyDAmonitor.diags.get_data() 
    metadata: metadata dict returned by Conventional.metadata from either ges or anl file
    '''
    
    # print(len(args))
    
    # check if the dfs passed are contain wind, which would have speed and direction
    if(metadata['Variable'] == 'uv'):
        print("Error: use wind_base_plots for wind datasets")
        return None
    
    #if variable is temperature, convert to F from K
    if(metadata['Variable'] == 't'):
        df_anl['observation'] = (1.8 * (df_anl['observation'] - 273.15)) + 32
        df_anl['omf_adjusted'] = 1.8 * df_anl['omf_adjusted']
        df_ges['observation'] = (1.8 * (df_ges['observation'] - 273.15)) + 32
        df_ges['omf_adjusted'] = 1.8 * df_ges['omf_adjusted']
        
    #* Create the bar plot by obs type showing proportional assimilated and total assimilated
    assimilated_by_obs_plots(df_anl)
    
    #*Plot histograms of obs, omf, and omb
    obs = df_ges['observation']
    omf = df_ges['omf_adjusted']
    oma = df_anl['omf_adjusted']
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    #TODO Add customized labels based on variable type
    data_list = [(obs, 'Observation', 'Observation Histogram'),
                (omf, 'Observation - Forecast', 'OMF'),
                (oma, 'Observation - Analysis', 'OMA')]

    for ax, (data, xlabel, title) in zip(axes, data_list):
        # Get basic statistics
        n, mean, std, mx, mn = len(data), np.mean(data), np.std(data), np.max(data), np.min(data)
        print(f'{title} Statistics: \nn: {n}, mean: {mean}, std: {std}, max: {mx}, min: {mn}\n')
        
        # Make proper bin sizes using the equation max-min/sqrt(n). Then
        # extend the bin range to 4x the standard deviation
        binsize = (mx - mn) / np.sqrt(n)
        bins = np.arange(mean - (4 * std), mean + (4 * std), binsize)
        # Plot histogram
        ax.hist(data, bins=bins)
        # Add labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(title + " Histogram", fontsize=14)

    # Add some more space between subplots
    plt.subplots_adjust(wspace=0.3)
    plt.show()
    
    #* Create spatial plots of obs, omf, and oma
    #get lats and lons
    latlons_ges = (df_ges['latitude'], df_ges['longitude'])
    latlons_anl = (df_anl['latitude'], df_anl['longitude'])
    
    #? what is the best way to handle having small scale base_plots and large scale base_plots?
    
    area_size = (latlons_ges[0].max() - latlons_ges[0].min()) * (latlons_ges[1].max() - latlons_ges[1].min())
    
    if(area_size<50): #(50 is arbitrary, meant to signify "zoomed in")
        #create small scale spatial plots 
        
        #create obs spatial plot
        fig, ax = plt.subplots(1, 1, figsize=(8,10), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=260)})
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.STATES, edgecolor='black')
        # ax.set_extent([-180, 180, -90, 90])
        # Plot the scatter data with smaller and more transparent points
        cs = ax.scatter(latlons_ges[1], latlons_ges[0], c=obs, s=20, cmap='Reds', alpha=0.7, transform=ccrs.PlateCarree())
        # Add gridlines for latitude and longitude
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        # Add a colorbar
        cb = plt.colorbar(cs, ax=ax, shrink=0.3, pad=.04, extend='both')
        cb.set_label('Observations')
        # Set title
        ax.set_title('Observations Map', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        #create spatial plot with omf and oma
        fig, ax = plt.subplots(1, 1, figsize=(8,10), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=260)})
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.STATES, edgecolor='black')
        # ax.set_extent([-180, 180, -90, 90])
        # Normalization for colorbar
        norm = mcolors.TwoSlopeNorm(vmin = 0 - np.std(omf), vcenter=0, vmax = 0 + np.std(omf))
        # # Plot the omf/oma data with two circle of different sizes (Option 1)
        cs1 = ax.scatter(latlons_ges[1], latlons_ges[0], c=omf, s=50, cmap='PRGn', transform=ccrs.PlateCarree(),
                         edgecolors='black', linewidths=0.2, label='omf', norm=norm)
        
        cs2 = ax.scatter(latlons_anl[1], latlons_anl[0], c=oma, s=10, cmap='PRGn', transform=ccrs.PlateCarree(),
                         edgecolors='black', linewidths=0.2, label='oma', norm=norm)
        # Plot the omf/oma data with two sqaures next to each other (Option 2)
        # lat_range = latlons_ges[0].max() - latlons_ges[0].min()
        # offset_ratio = 80
        # offset = lat_range / offset_ratio
        # cs1 = ax.scatter(latlons_ges[1], latlons_ges[0]+offset, c=omf, s=30, cmap='PRGn', marker='s', 
        #                  edgecolors='black', linewidths=0.7, transform=ccrs.PlateCarree(), label='omf', norm=norm)
        # cs2 = ax.scatter(latlons_anl[1], latlons_anl[0]-offset, c=oma, s=30, cmap='PRGn', marker='s', 
        #                  edgecolors='black', linewidths=0.7, transform=ccrs.PlateCarree(), label='oma', norm=norm)
        # Add gridlines for latitude and longitude
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        # Add a colorbar adn legend
        cb = plt.colorbar(cs2, ax=ax, shrink=0.3, pad=.04, extend='both')
        cb.set_label('OmF and OmA')
        ax.legend()
        # Set title
        ax.set_title('OmF and OmA Comparison', fontsize=14)
        plt.tight_layout()
        plt.show()
        
    else:
        # Create large scale spatial subplots
        fig, axes = plt.subplots(3, 1, figsize=(15,20), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=260)})
        
        data_list = [(obs, 'Observation', 'Observations Map', 'Reds', latlons_ges),
                    (omf, 'OmF', 'OmF Map', 'PRGn', latlons_ges),
                    (oma, 'OmA', 'OmA Map', 'PRGn', latlons_anl)]
        
        #? Add extra dataset here to data_list if there are more than 2?

        #Plot each of the datasets spatially, obs, oma, and omf
        for ax, (data, label, title, color, coords) in zip(axes, data_list):
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
            ax.add_feature(cfeature.STATES, edgecolor='black')
            # ax.set_extent([-180, 180, -90, 90])
            # Normalization for diverging cmaps, none for increasing cmaps for obs map
            if(label == "Observation"):
                norm=None
            else:
                norm = mcolors.TwoSlopeNorm(vmin = 0 - (np.std(data)*2), vcenter=0, vmax = 0 + (np.std(data)*2))
            # Plot the scatter data with smaller and more transparent points
            cs = ax.scatter(coords[1], coords[0], c=data, s=20, cmap=color, alpha=0.7, transform=ccrs.PlateCarree(), norm=norm)
            # Add gridlines for latitude and longitude
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0)
            gl.top_labels = False
            gl.right_labels = False
            # Add a colorbar
            cb = plt.colorbar(cs, ax=ax, shrink=0.3, pad=.04, extend='both')
            cb.set_label(label)
            # Set title
            ax.set_title(title, fontsize=14)
            
        plt.tight_layout()
        plt.show()
        
        
def wind_base_plots(df_anl, df_ges, metadata, **args):
    
    # check if the dfs passed are contain wind, which would have speed and direction
    if(metadata['Variable'] != 'uv'):
        print("Error: use base_plots for non wind datasets")
        return None
    
    # Get data arrays
    u_obs = df_ges['u_observation']
    v_obs = df_ges['v_observation']
    u_omf = df_ges['u_omf_adjusted']
    v_omf = df_ges['u_omf_adjusted']
    u_oma = df_anl['u_omf_adjusted']
    v_oma = df_anl['u_omf_adjusted']
    
    #* Create the bar plot by obs type showing proportional assimilated and total assimilated
    assimilated_by_obs_plots(df_anl)
    
    #*Plot histograms of obs, omf, and omb
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 5))

    #TODO Add customized labels based on variable type
    data_list = [(u_obs, 'U Observation', 'U Observation Histogram'),
                 (u_omf, 'U OmF', 'U OmF Histogram'),
                 (u_oma, 'U OmA', 'U OmA Histogram'),
                 (v_obs, 'V Observation', 'V Observation Histogram'),
                 (v_omf, 'V OmF', 'V OmF Histogram'),
                 (v_oma, 'V OmA', 'U OmA Histogram')]

    axes = axes.flatten()
    for ax, (data, xlabel, title) in zip(axes, data_list):
        # Get basic statistics
        n, mean, std, mx, mn = len(data), np.mean(data), np.std(data), np.max(data), np.min(data)
        print(f'{title} Statistics: \nn: {n}, mean: {mean}, std: {std}, max: {mx}, min: {mn}\n')
        
        # Make proper bin sizes using the equation max-min/sqrt(n). Then
        # extend the bin range to 4x the standard deviation
        binsize = (mx - mn) / np.sqrt(n)
        bins = np.arange(mean - (4 * std), mean + (4 * std), binsize)
        # Plot histogram
        ax.hist(data, bins=bins)
        # Add labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(title + " Histogram", fontsize=14)

    # Add some more space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.show()
    
    #* Create spatial plots of obs, omf, and oma
    
    #get lats and lons
    latlons_ges = (df_ges['latitude'], df_ges['longitude'])
    latlons_anl = (df_anl['latitude'], df_anl['longitude'])
    
    #? what is the best way to handle having small scale base_plots and large scale base_plots?
    area_size = (latlons_ges[0].max() - latlons_ges[0].min()) * (latlons_ges[1].max() - latlons_ges[1].min())
    
    # Create the plot of observations
    plt.figure(figsize=(15, 12))
    
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
    ax.add_feature(cfeature.STATES, edgecolor='black')
    # ax.set_extent([-180, 180, -90, 90])
    # Plot the scatter data with smaller and more transparent points
    mag = np.sqrt( (u_obs**2) + (v_obs**2) )
    u_norm = u_obs / mag
    v_norm = v_obs / mag
    cs = plt.quiver(latlons_ges[1], latlons_ges[0], u_norm, v_norm, mag, scale=50, cmap='plasma', transform=ccrs.PlateCarree())
    # Add a colorbar
    cb = plt.colorbar(cs, shrink=0.5, pad=.04, extend='both')
    cb.set_label('Wind Speed')
    # Add gridlines for latitude and longitude
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    plt.title("Wind Speed and Direction Observations")
    # Display the plot
    plt.show()
    
def assimilated_by_obs_plots(df_anl):
    
    #* Create the bar plot by obs type showing proportional assimilated and total assimilated
    
    #calculated prop and totals
    prop_assim_by_obs_type_anl = pd.DataFrame(df_anl.groupby('observation_type')['analysis_use_flag'].mean())
    total_assim_by_obs_type_anl = pd.DataFrame(df_anl.groupby('observation_type')['analysis_use_flag'].sum())
    
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    
    #create bars for prop
    axes[0].bar(range(len(prop_assim_by_obs_type_anl['analysis_use_flag'])),
                prop_assim_by_obs_type_anl['analysis_use_flag'],
                tick_label=prop_assim_by_obs_type_anl.index)
    # Add labels and title as necessary
    axes[0].set_xlabel('Obs Type')
    axes[0].set_ylabel('Prop. Assimilated')
    axes[0].set_title('Proportion Assimilated Obs by Obs Type')
    #create bars for prop
    axes[1].bar(range(len(total_assim_by_obs_type_anl['analysis_use_flag'])),
                total_assim_by_obs_type_anl['analysis_use_flag'],
                tick_label=prop_assim_by_obs_type_anl.index)
    # Add labels and title as necessary
    axes[1].set_xlabel('Obs Type')
    axes[1].set_ylabel('Total Assimilated')
    axes[1].set_title('Total Assimilated Obs by Obs Type')
        
    plt.show()
    