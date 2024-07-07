#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:05:35 2023

@author: bonnie
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from pathlib import Path
import numpy as np
import glob
import tables
import warnings
from tierpsytools.analysis.count_worms import n_worms_per_frame
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata
from tierpsytools.hydra.match_wells_annotations import print_elements_in_a_not_in_b

CUSTOM_STYLE = '/Users/bonnie/Disease-Modelling/gene_cards.mplstyle'
# CUSTOM_STYLE = '/Volumes/behavgenom$/Bonnie/Scripts/Disease-Modelling/gene_cards.mplstyle'

#%%
def wells_not_tracked(feat_file, fname_file, meta_file, save_to, group_by):
    
    """
    
    Finds wells that have not been tracked i.e. exist in the metadata 
    but have not had features extracted
    
    Parameters
    ----------
    feat_file : file path to tierpsy features summaries file
        File must have a file_id and well_id column.
    fname_file : file path to tierpsy filenames summaries file.
        File must have a file_id and filename column.
    meta_file : file path to metadata file
    group_by: string
        Variable e.g. drug, bacterial strain
    save_to : file path
    
    Returns
    -------
    grouped: 
        Dataframe containing n_wells not tracked per variable in group_by
    
    """
    
    # Creates matching features and metadata dfs from the .csv files 
    feat, meta = read_hydra_metadata(feat_file, fname_file, meta_file)
    
    # Load original meta
    concat_meta = pd.read_csv(meta_file)
    
    # check group_by column is a string as can only concatenate str (not "int") to str
    meta[group_by] = meta[group_by].astype("string")
    concat_meta[group_by] = concat_meta[group_by].astype("string")
    
    # Create 'compare' column to find differences in meta and original meta 
    # Imgstore_name and well_name are needed for checking videos in tierpsy viewer    
    concat_meta['compare'] = concat_meta['imgstore_name'] + '/_' + concat_meta[
        'well_name'] + '_' + concat_meta[group_by]
    
    meta['compare'] = meta['imgstore_name'] + '/_' + + meta[
        'well_name'] + '_' + meta[group_by]
    
    dif_list = pd.DataFrame([x for x in list(concat_meta['compare'].unique(
        )) if x not in list(meta['compare'].unique())])
    
    dif_list[group_by] = dif_list[0].str.split('_').str[-1]
    dif_list['well_name'] = dif_list[0].str.split('_').str[-2]
    dif_list['imgstore_name'] = dif_list[0].str.split('/').str[-2]
    dif_list = dif_list.drop(columns=0)
    
    # Find number of wells not tracked per variable
    grouped = pd.DataFrame(dif_list.groupby([group_by]).size(
        )).sort_values(by=0,ascending=False)
    
    print('The number of wells not tracked is', dif_list.shape[0], '/'
          ,concat_meta.shape[0])
    
    dif_list.to_csv(save_to / 'missing_wells.csv', index=False)
    
    return grouped

#%%
def get_videos(meta, variables_list, column, figures_dir):
    
    """
    
    Finds video files to check in tierpsy viewer
    
    Parameters
    ----------
    meta: dataframe shape = (n_wells, n_meta_cols)
        Metadata that match the features dataframe row-by-row
    variables_list: list or dictionary
        Variables (e.g. drug, bacterial strain) of interest
    column : str
        Metadata column that variables_list corresponds to
    
    Returns
    -------
    None.
    
    """
    
    for v in variables_list:
        df = meta[meta[column]==v]
        
        # select columns needed for tierpsy viewer
        df = df[['imgstore_name_bluelight','imgstore_name_poststim', 
                  'imgstore_name_prestim', 'well_name', column]]
        
        df['imgstore_name_bluelight'] = df['imgstore_name_bluelight'].str.split('/').str[1]
        df['imgstore_name_prestim'] = df['imgstore_name_prestim'].str.split('/').str[1]
        df['imgstore_name_poststim'] = df['imgstore_name_poststim'].str.split('/').str[1]
        
        # df = df[['imgstore_name','well_name', column]]
        
        # add column for comments
        df['comment'] = ''
        
        # Set save path
        saveto = figures_dir / v
        saveto.mkdir(exist_ok=True)
        
        df.to_csv(saveto / 'video_files.csv')
    
    return

#%%
def basic_plot(meta, feat, x_order, x, ft, hue='date_yyyymmdd', 
               title=None, stats=None, save_to=None):
    
    """
    
    Plots 
    
    Parameters
    ----------
    feat: dataframe shape = (n_wells, n_features * n_bluelight_conditions)
        The features at every bluelight condition for each well.
    meta: dataframe shape = (n_wells, n_meta_cols)
        The metadata that match the features dataframe row-by-row
    x_order: list
        List of variables, e.g. control and drug, in correct order to plot 
    x: str
        Metadata column that corresponds to x_order
    ft: str 
        Tierpsy feature
    hue: str
        Metadata column to colour individual points of swarmplot by 
        The default is 'date_yyyymmdd'
    title: str, optional
        The default is None
    stats: dataframe shape = (n_features), optional
        P-values 
    save_to: str, optional
        
    Returns
    -------

    ax: figure
    
    """
    
    data = pd.concat([meta, feat], axis=1) 
    
    data = data[data[x].isin(x_order)]
        
    label_format = '{0:.4g}'
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    plt.tight_layout()
    
    plt.figure(figsize=(6,10))
    ax = sns.boxplot(y=ft,
                x=x,
                order = x_order,
                data=data,              
                color = 'lightgrey',
                showfliers=False)
    plt.tight_layout()
       
    ax = sns.swarmplot(y=ft,
                x=x, 
                order = x_order,
                data=data,
                hue = hue,
                palette='mako',
                alpha=1)
    
    ax.set_ylabel(fontsize=22, ylabel=ft)
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])#, fontsize=16) #labels = ax.get_yticks(),
    
    # ax.set_xticklabels(labels=x_order,rotation=90)
    
    plt.legend(loc='upper right')
    plt.legend(title = hue, title_fontsize = 14,fontsize = 14, 
                bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    if title is not None:
        plt.title(title)
    
    if stats is not None:
        
        if type(stats) == str:
            stats = pd.read_csv(stats)
       
        add_stat_annotation(ax,
                                data=data,
                                x=x,
                                y=ft,
                                order=x_order,
                                box_pairs=[x_order],
                                perform_stat_test=False,
                                pvalues=[stats[ft].values[0]], #pvalues=[bhP_values_df[feature].values[0]],
                                test=None,
                                # text_format='star',
                                loc='outside',
                                verbose=2,
                                text_annot_custom=[
                                    'p={:.3E}'.format(
                                        round(stats[ft].values[0],100))], #:.4f
                                fontsize=20,
                                )
        plt.tight_layout()
    
    if save_to is not None:
            # cg.savefig(Path(saveto) / '{}_clustermap_dpi1000_resolution.png'.format(stim), dpi=1000)
            plt.savefig(Path(save_to) / '{}.png'.format(ft), dpi=300, bbox_inches='tight')
    
    return ax

#%%
def n_worms_per_well(results_folder):
    
    """
    
    Parameters
    ----------
    Path: path
        Path to day Results folder
    
    Returns
    -------
    
    """
    
    results_folder_str = str(results_folder)
        
    filenames = glob.glob(results_folder_str + "/**/metadata_featuresN.hdf5", recursive = True)
    
    li = []
    # for filename in filenames:
    for count, filename in enumerate(filenames):
        print('Analysing {} {}/{}'.format(filename, count+1, len(filenames)))    
        
        # Open the file in read mode
        h5file = tables.open_file(filename, 'r')
        
        # Access the trajectories data
        trajectories = h5file.root.trajectories_data
        
        # Read in frame_number column
        trajectories_data = pd.DataFrame({col: trajectories.col(col) for col in ['frame_number']})
        
        # Find frame number for video
        # poststim/prestim = 7500, bluelight = 9000
        max_t = trajectories_data['frame_number'].max()
        
        # Access the timeseries table
        timeseries = h5file.root.timeseries_data
        
        # Select columns
        columns_to_read = ['worm_index','well_name', 'timestamp']
        
        # Read the columns
        timeseries_data = {col: timeseries.col(col) for col in columns_to_read}
        
        # Close the file
        h5file.close()
        
        # convert to dataframe
        worms = pd.DataFrame(timeseries_data)
        
         # convert well name to usable format
        worms['well_name'] = np.vstack(timeseries_data['well_name']).astype(np.str)
    
        # get number of worms per frame per well
        n_worms = pd.DataFrame(
            worms.groupby('well_name')['timestamp'].apply(n_worms_per_frame, max_t = max_t))

        # get average number of worms per frame per well
        n_worms_well = n_worms.groupby('well_name').mean()
        n_worms_well = n_worms_well.rename(columns=({'timestamp':'n_worms_mean'}))
        
        # get number of worms in the well
        n_worms_well['n_worms'] = pd.DataFrame(n_worms.groupby('well_name').max())
        
        # get imgstore_name for merging with metadata
        file_name = filename.split('/')
        n_worms_well['imgstore_name'] = '/'.join([file_name[i] for i in [-3, -2]])
        
        li.append(n_worms_well)

    all_worms = pd.concat(li, axis=0, ignore_index=False)
    
    saveto = Path(results_folder_str.replace('Results','AuxiliaryFiles'))
    all_worms.to_csv(saveto / 'n_worms.csv',
                     index = True)
    
    return all_worms 

#%%

# def update_metadata_with_worms(
#         day_metadata_file, worm_file,
#         merge_on=['imgstore_name', 'well_name'],
#         ):
            
#     metadata = pd.read_csv(day_metadata_file)
#     worm_metadata = pd.read_csv(worm_file)

#     print(f'shape of metadata before merging: {metadata.shape}')
#     print(f'shape of worms before merging: {worm_metadata.shape}')

#     worm_updated_metadata = pd.merge(
#          metadata, worm_metadata, on=merge_on,
#         how='outer')
    
#     worm_updated_metadata[['n_worms_mean','n_worms']] = worm_updated_metadata[
#         ['n_worms_mean','n_worms']].fillna(0)
    
#     print(f'shape of annotated metadata: {worm_updated_metadata.shape}')

#     worm_updated_metadata.to_csv(day_metadata_file, index=False)

#     return worm_updated_metadata

#%%
def update_metadata_with_worms_annotations(
        aux_dir, n_wells=96, saveto=None, del_if_exists=True):
    
    from tierpsytools.hydra.match_bluelight_videos import (
        match_bluelight_videos_in_folder)
    from tierpsytools.hydra.hydra_helper import explode_df
    
    # input check
    if isinstance(aux_dir, str):
        aux_dir = Path(aux_dir)
    if saveto is None:
        saveto = aux_dir / 'worm_updated_metadata.csv'
    elif isinstance(saveto, str):
        saveto = Path(saveto)

    # check if destination file exists
    if saveto.exists():
        warnings.warn(
            f'Metadata with worms, {saveto}, already exists.')
        if del_if_exists:
            warnings.warn('File will be overwritten.')
            saveto.unlink()
        else:
            warnings.warn(
                'Nothing to do here. If you want to recompile the day metadata'
                ', rename or delete the existing file.')
            # return
    
    # load wells_updated metadata, worm metadata, and matched rawvideos data

    # find metadata, checks
    metadata_fname = list(aux_dir.rglob('wells_updated_metadata.csv'))
    if len(metadata_fname) > 1:
        warnings.warn(
            f'More than one metadata file in {aux_dir}: \n' +
            f'{metadata_fname} \naborting.')
        # return
    elif len(metadata_fname) == 0:
        warnings.warn(f'no metadata file in {aux_dir}, aborting.')
        # return
    else:
        metadata_fname = metadata_fname[0]
    
    # load metadata
    metadata_df = pd.read_csv(metadata_fname)
    if metadata_df['imgstore_name'].isna().any():
        warning_msg = (
            f"There are {metadata_df['imgstore_name'].isna().sum()}"
            + ' NaN values in the `imgstore_name column` in the metadata.\n'
            + 'If this is unexpected, you should check your metadata.'
            )
        warnings.warn(warning_msg)
        metadata_df = metadata_df.dropna(subset=['imgstore_name'])
    
    # find all day worms files in aux_dir
    worms_files = list(Path(aux_dir).rglob('*n_worms.csv'))

    # loop, read each annotation as a dataframe, concatenate as a single df
    worms_df=[]
    for f in worms_files:
        worms = pd.read_csv(f)
        worms_df.append(worms)
    worms_df = pd.concat(worms_df,axis=0,ignore_index=True)
    
    # strip the imgstore of the date_yyyymmdd/ part
    worms_df.loc[:, 'imgstore'] = worms_df['imgstore_name'].apply(
        lambda x: x.split('/')[1])
 
    # get matched rawvideos names 
    raw_dir = Path(str(aux_dir).replace('AuxiliaryFiles', 'RawVideos'))
    rawvids = match_bluelight_videos_in_folder(raw_dir)
    
#%%
    # TODO: Hardcode
    # Get dataframe of channels and wells
    CH2W_df = pd.DataFrame(columns=['channel', 'well_name'])
    
    if n_wells == 24:
    
        CH2W = {
            'Ch1': ['A1', 'B1', 'A2', 'B2'],
            'Ch2': ['C1', 'D1', 'C2', 'D2'],
            'Ch3': ['A3', 'B3', 'A4', 'B4'],
            'Ch4': ['C3', 'D3', 'C4', 'D4'],
            'Ch5': ['A5', 'B5', 'A6', 'B6'],
            'Ch6': ['C5', 'D5', 'C6', 'D6']
        }
    
    elif n_wells == 96:
        
        CH2W = {
            'Ch1': ['A1', 'B1', 'C1','D1','A2', 'B2', 'C2', 'D2', 'A3', 'B3', 'C3', 'D3','A4', 'B4', 'C4', 'D4'],
            'Ch2': ['E1', 'F1', 'G1','H1', 'E2', 'F2', 'G2', 'H2', 'E3', 'F3', 'G3', 'H3', 'E4', 'F4', 'G4', 'H4'],
            'Ch3': ['A5', 'B5', 'C5', 'D5', 'A6', 'B6', 'C6', 'D6', 'A7', 'B7', 'C7', 'D7', 'A8', 'B8', 'C8', 'D8'],
            'Ch4': ['E5', 'F5', 'E6', 'F6', 'G6', 'H6', 'E7', 'F7', 'G7', 'H7', 'E8', 'F8', 'G8', 'H8'],
            'Ch5': ['A9', 'B9', 'C9', 'D9', 'A10', 'B10', 'C10', 'D10', 'A11', 'B11', 'C11','D11', 'A12', 'B12', 'C12', 'D12'],
            'Ch6': ['E9', 'F9', 'G9', 'H9', 'E10', 'F10', 'G10', 'H10', 'E11', 'F11', 'G11','H11', 'E12', 'F12', 'G12', 'H12']
            }
        
    elif n_wells == 6:
        
        CH2W = {
                'Ch1':['A1'],
                'Ch2':['B1'],
                'Ch3':['A2'],
                'Ch4':['B2'],
                'Ch5':['A3'],
                'Ch6':['B3']
                }
    
    # Populate the DataFrame
    for channel, well_name in CH2W.items():
        CH2W_df = CH2W_df.append({'channel': channel, 'well_name': well_name
                                  }, ignore_index=True)
      
    # expand dataframe to have 1 row per well
    rawvids = pd.merge(rawvids, CH2W_df)
    rawvids = explode_df(rawvids, 'well_name')
    
    # assert rawvids.shape[0] * 3 == metadata_df.shape[0]
    
#%%
    conditions = ['prestim', 'bluelight', 'poststim']
    
    # Columns to add the suffix to for each condition
    columns_to_rename = {
        'prestim': ['imgstore', 'n_worms'],
        'bluelight': ['imgstore', 'n_worms'],
        'poststim': ['imgstore', 'n_worms']
    }
    
    # DataFrames to store the results for each condition
    dfs_by_condition = {}
    
    # Merge columns for each condition
    for condition in conditions:
        # Create boolean mask for the condition
        condition_mask = worms_df['imgstore'].str.contains(
            condition, case=False, na=False)
        
        # Create DataFrame based on the mask
        condition_df = worms_df[condition_mask].copy()
        
        # Rename specified columns
        condition_df = condition_df.rename(columns=lambda x: 
                                           x + f'_{condition}' if x in 
                                           columns_to_rename[condition] else x)
        
        # Store the DataFrame
        dfs_by_condition[condition] = condition_df
    
    # Merge the DataFrames
    matched_df = pd.merge(rawvids, dfs_by_condition['prestim'], 
                          on=['well_name', 'imgstore_prestim'])
    
    matched_df = pd.merge(matched_df, dfs_by_condition['bluelight'], 
                          on=['well_name', 'imgstore_bluelight'])
    
    matched_df = pd.merge(matched_df, dfs_by_condition['poststim'], 
                          on=['well_name', 'imgstore_poststim'])
    
#%%     
    # get maximum number of worms in a well in any of the  3 conditions
    matched_df['n_worms'] = matched_df[
        ['n_worms_prestim', 'n_worms_bluelight', 'n_worms_poststim']].max(axis=1)
    
    # select relevant columns only
    matched_df_new = matched_df[
        ['imgstore_bluelight','imgstore_prestim','imgstore_poststim',
         'well_name','n_worms']]
    
    # re-assign maximum number of worms across all 3 conditioons
    new_worms_df = pd.DataFrame(
        matched_df_new[
            ['imgstore_bluelight', 'imgstore_prestim', 'imgstore_poststim']
            ].stack().reset_index(level=1, drop=True), columns=['imgstore'])
    
    new_worms_df['well_name'] = matched_df.loc[new_worms_df.index, 'well_name']
    new_worms_df['n_worms'] = matched_df.loc[new_worms_df.index, 'n_worms']

    new_worms_df.reset_index(drop=True, inplace=True)
    
    # merge with metadata
    worms_updated_metadata = pd.merge(metadata_df,new_worms_df, 
                                      on=['well_name','imgstore'],
                                      how='left')
    
    worms_updated_metadata['n_worms'] = worms_updated_metadata['n_worms'].fillna(0)
    
    assert worms_updated_metadata.shape[0] == metadata_df.shape[0]
    
    worms_updated_metadata.to_csv(saveto, index=False)
    
#%%
def plot_worms(meta, variables, column, figures_dir):
    
    """
    Parameters
    ----------
    meta_file: dataframe shape = (n_wells, n_meta_cols)
        Worms_updated_metadata file
    variables: list
        List of variables, e.g. control and drug
    column: str
        Metadata column that corresponds to variables
    figures_dir: str, optional
        
    Returns
    -------
    
    """
    
    # meta = pd.read_csv(meta_file)
   
    li = []
    for v in variables:
        day_meta = meta[meta[column]==v]
        
        grouped = pd.DataFrame(day_meta.groupby('n_worms').size())
        
        # Calculate the total sum of the 'Value' column
        total_sum = grouped[0].sum()
        
        # Create a new column 'Percentage' with calculated percentages
        grouped['Percentage'] = (grouped[0] / total_sum) * 100
        
        grouped[column] = v
        
        li.append(grouped)
    
    percentages = pd.concat(li)
    
    # Plot histogram
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    plt.tight_layout()
    
    sns.barplot(data=percentages, x=percentages.index, y='Percentage', hue=column)
    plt.xticks(fontsize=8)
    plt.xlabel('Worms per well')
    
    # TODO: string or Path? 
    plt.savefig(figures_dir / 'n_worms.png', bbox_inches='tight',dpi=300)

#%%

def average_control(feat_df, meta_df, group_by='date_yyyymmdd'):
    """ Average data for control plate on each experiment day to yield a single mean datapoint for
        the control. This reduces the control sample size to equal the test strain sample size, for
        t-test comparison. Information for the first well in the control sample on each day is used
        as the accompanying metadata for mean feature results.
       
        Input
        -----
        features, metadata : pd.DataFrame
            Feature summary results and metadata dataframe with multiple entries per day
           
        Returns
        -------
        features, metadata : pd.DataFrame
            Feature summary results and metadata with control data averaged (single sample per day)
    """
    # Take mean of control for each plate = collapse to single datapoint
    mean_control = meta_df[[group_by]].join(feat_df).groupby(
                                    by=[group_by]).mean().reset_index()
   
    # Append remaining control metadata column info (with first well data for each date)
    remaining_cols = [c for c in meta_df.columns.to_list()
                      if c not in [group_by]]
    
    mean_control_row_data = []
    for i in mean_control.index:
        groupby = mean_control.loc[i, group_by]
        control_date_meta = meta_df.loc[meta_df[group_by] == groupby]

        first_well = control_date_meta.loc[control_date_meta.index[0], remaining_cols]
        first_well_mean = first_well.append(mean_control.loc[mean_control[group_by] == groupby
                                                             ].squeeze(axis=0))
        mean_control_row_data.append(first_well_mean)
   
    control_mean = pd.DataFrame.from_records(mean_control_row_data)
    meta_df = control_mean[meta_df.columns.to_list()]
    feat_df = control_mean[feat_df.columns.to_list()]
    
    return feat_df, meta_df

    # feat = pd.concat([feat.loc[meta['drug_type'] != "NoCompound", :],
    #                       control_features], axis=0).reset_index(drop=True)       
    # meta = pd.concat([meta.loc[meta['drug_type'] != "NoCompound", :],
    #                       control_metadata.loc[:, meta.columns.to_list()]],
    #                       axis=0).reset_index(drop=True)
