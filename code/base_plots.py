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
        plt.text(bar.get_x() + bar.get_width() / 2, height - 0.01, f'{'Total: ' + str(total_count)}', 
                ha='center', va='top', rotation=90)
        
    plt.show()
    
    
    #*Plot histograms of obs, omf, and omb
    
    obs = df_anl['observation']
    omf_ges = df_ges['omf_adjusted']
    omf_anl = df_anl['omf_adjusted']
   
    fig, axes = plt.subplots(1, 3, figsize = (16,5))
    
    #Plot Observations
    
    # Get basic statistics
    n, mean, std, mx, mn = len(obs), np.mean(obs), np.std(obs), np.max(obs), np.min(obs)
    print(f'Observation Statistics: \nn: {n}, mean: {mean}, std: {std}, max: {mx}, min: {mn}\n')

    # Make proper bin sizes using the equation max-min/sqrt(n). Then
    # extend the bin range to 4x the standard deviation
    binsize = (mx-mn)/np.sqrt(n)
    bins = np.arange(mean-(4*std),mean+(4*std),binsize)
    
    axes[0].hist(obs, bins=bins)

    # Add labels
    axes[0].set_xlabel('Observation')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Observation Histogram', fontsize=14) #TODO Add customized labels based on the data file
    
    #Plot omf_anl
    
    # Get basic statistics
    n, mean, std, mx, mn = len(omf_ges), np.mean(omf_ges), np.std(omf_ges), np.max(omf_ges), np.min(omf_ges)
    print(f'OMF Analysis Statistics: \nn: {n}, mean: {mean}, std: {std}, max: {mx}, min: {mn}\n')

    # Make proper bin sizes using the equation max-min/sqrt(n). Then
    # extend the bin range to 4x the standard deviation
    binsize = (mx-mn)/np.sqrt(n)
    bins = np.arange(mean-(4*std),mean+(4*std),binsize)

    axes[1].hist(omf_ges, bins=bins)

    # Add labels
    axes[1].set_xlabel('Observation - Forecast Guess')
    axes[1].set_ylabel('Count')
    axes[1].set_title('OMF Guess Histogram', fontsize=14)
    
    #Plot omf_ges
    
    # Get basic statistics
    n, mean, std, mx, mn = len(omf_anl), np.mean(omf_anl), np.std(omf_anl), np.max(omf_anl), np.min(omf_anl)
    print(f'OMF Analysis Statistics: \nn: {n}, mean: {mean}, std: {std}, max: {mx}, min: {mn}\n')

    # Make proper bin sizes using the equation max-min/sqrt(n). Then
    # extend the bin range to 4x the standard deviation
    binsize = (mx-mn)/np.sqrt(n)
    bins = np.arange(mean-(4*std),mean+(4*std),binsize)

    axes[2].hist(omf_anl, bins=bins)

    # Add labels
    axes[2].set_xlabel('Observation - Forecast Analysis')
    axes[2].set_ylabel('Count')
    axes[2].set_title('OMF Analysis Histogram', fontsize=14)
    
    #add some more space between subplots
    plt.subplots_adjust(wspace=0.3)

    plt.show()