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

def base_plots(df_anl, df_ges):

    '''
    df_anl : pd dataframe returned by pyDAmonitor.diags.get_data() 
    df_ges : pd dataframe returned by pyDAmonitor.diags.get_data() 
    '''
    
    
    
    #* Create the bar plot by obs type showing proportional assimilated and total assimilated
    
    #calculated prop and totals
    prop_assim_by_obs_type_anl = pd.DataFrame(df_anl.groupby('observation_type')['analysis_use_flag'].mean())
    total_assim_by_obs_type_anl = pd.DataFrame(df_anl.groupby('observation_type')['analysis_use_flag'].sum())
    
    #create bars
    bars = plt.bar(range(len(prop_assim_by_obs_type_anl['analysis_use_flag'])),
                prop_assim_by_obs_type_anl['analysis_use_flag'],
                tick_label=prop_assim_by_obs_type_anl.index)

    # Add labels and title as necessary
    plt.xlabel('Obs Type')
    plt.ylabel('Prop. Assimilated')
    plt.title('Proportional and Total Assimilated Obs by Obs Type')

    # Add the total count on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        total_count = total_assim_by_obs_type_anl['analysis_use_flag'].iloc[i].astype(int)
        plt.text(bar.get_x() + bar.get_width() / 2, height - 0.01, 'Total: ' + str(total_count), 
                ha='center', va='top', rotation=90)
        
    plt.show()
    
    
    #*Plot histograms of obs, omf, and omb
    
    obs = df_ges['observation']
    omf_ges = df_ges['omf_adjusted']
    omf_anl = df_anl['omf_adjusted']
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    #TODO Add customized labels based on variable type
    data_list = [(obs, 'Observation', 'Observation Histogram'),
                (omf_ges, 'Observation - Forecast Guess', 'OMF Guess Histogram'),
                (omf_anl, 'Observation - Forecast Analysis', 'OMF Analysis Histogram')]

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
        ax.set_title(title, fontsize=14)

    # Add some more space between subplots
    plt.subplots_adjust(wspace=0.3)

    plt.show()

    
    #* Create spatial plots of obs, omf_ges, and omf_anl
    
    #get lats and lons
    latlons_ges = (df_ges['latitude'], df_ges['longitude'])
    latlons_anl = (df_anl['latitude'], df_anl['longitude'])
    ''' 
    #! we have to do this^ because for some reason there are unexpected
    #! differences in the _anl and _ges file so we can get slightly 
    #! numbers of obs when filtering
    '''
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(15,20), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    
    data_list = [(obs, 'Observation', 'Observations Map', 'Reds', latlons_ges),
                (omf_ges, 'OmF Guess', 'OMF Guess Map', 'PRGn', latlons_ges),
                (omf_anl, 'OmF Analysis', 'OMF Analysis Map', 'PRGn', latlons_anl)]

    for ax, (data, label, title, color, coords) in zip(axes, data_list):
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
        ax.add_feature(cfeature.STATES, edgecolor='black')
        # ax.set_extent([-180, 180, -90, 90])
        
        # Plot the scatter data with smaller and more transparent points
        cs = ax.scatter(coords[1], coords[0], c=data, s=20, cmap=color, alpha=0.7, transform=ccrs.PlateCarree())
        
        # Add a colorbar
        cb = plt.colorbar(cs, ax=ax, shrink=0.5, pad=.04, extend='both')
        cb.set_label(label)
        
        # Set title
        ax.set_title(title, fontsize=14)
        
    plt.tight_layout()
        
    plt.show()
    
    