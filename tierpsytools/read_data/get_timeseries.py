#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:16:50 2019

@author: em812
"""
import pandas as pd
import numpy as np
from time import time

def get_timeseries(root_dir, select_keywords=None, drop_keywords=None,
                   names=None, only_wells=None):
    """
    Get timeseries data from *_featuresN files under a root directory.
    Input:
        root_dir = results root directory
        select_keywords = select *_featuresN.hdf5 files that contain any of
                            these keywords in the file path
        drop_keywords = ignore *_featuresN.hdf5 files that contain any of
                            these keywords in the file path
        names = timeseries names to exctact. If None, all timeseries data will
                be returned
        only_wells = list of well_names to read from the featuresN.hdf5 file.
                     If None, the timeseries will not be filtered by well
            (        good for legacy data)
    Return:
        filenames = dataframe with filenames and file_ids
        timeseries =
    """
    filenames = get_filenames(root_dir)

    filenames = select_filenames(filenames, select_keywords, drop_keywords)

    data = {}
    start_time = time()
    for ifl, (fileid, file) in enumerate(
            filenames[['file_id', 'file_name']].values):
        file_time = time()
        print('Reading timeseries from file {} of {}'.format(
            ifl+1, filenames.shape[0]))
        timeseries = read_timeseries(file, names=names, only_wells=only_wells)
        data[fileid] = timeseries
        print('File read in {} sec.'.format(time()-file_time))
    print('Done reading in {} sec.'.format(time()-start_time))

    return filenames, data

def read_timeseries(filename, names=None, only_wells=None):
    """
    Read timeseries from a *featuresN.hdf5 file from tierpsy.
    Input:
        filename = name of file
        names = names of timeseries to read.
                If none, all timeseries will be read
        only_wells = list of well_names to read from the featuresN.hdf5 file.
                     If None, the timeseries will not be filtered by well
                     (good for legacy data)
    """
    assert isinstance(only_wells, list), 'only_wells must be a list'
    assert all(isinstance(well, str) for well in only_wells), \
        'only_wells must be a list'
    with pd.HDFStore(filename, 'r') as f:
        if only_wells is None:
            series = f['timeseries_data']
        else:
            query_str = 'well_name in {}'.format(only_wells)
            series = f['timeseries_data'].query(query_str)
    if names is None:
        return series
    else:
        return series[names]


#%% helper functions
def select_filenames(filenames, select_keywords, drop_keywords):
    """
    Filter the filenames list based on keywords
    Input:
        filenames = a dataframe of filenames and file_ids
        select_keywords = list of strings. Files that contain any of
                        these strings in the relative file path
                        will be selected
        drop_keywords = list of strings. Files that contain any of
                        these strings in the relative file path
                        will be ignored
    """
    from os.path import relpath

    if select_keywords is None and drop_keywords is None:
        return filenames

    filenames['_relative_path'] = filenames['file_name'].apply(
        lambda x: relpath(x, root_dir))

    if select_keywords is not None:
        if isinstance(select_keywords, str):
            select_keywords = [select_keywords]

        keep = filenames['_relative_path'].apply(
            lambda x:
                True if np.any([kwrd in x for kwrd in select_keywords])
                else False)

        filenames = filenames.loc[keep, :]

    if drop_keywords is not None:
        if isinstance(select_keywords, str):
            drop_keywords = [drop_keywords]

        drop = filenames['_relative_path'].apply(
            lambda x:
                False if np.any([kwrd in x for kwrd in drop_keywords])
                else True)

        filenames = filenames.loc[drop, :]

    filenames.drop(columns=['_relative_path'])
    filenames.reset_index(drop=True)

    return filenames

def get_filenames(root_dir):
    """
    Find all *featuresN.hdf5 files under root_dir
    Input:
        root_dir = path to results root directory
    Return:
        filenames = dataframe listing *featuresN.hdf5 file paths and assigning
                    file_id to each file (tierpsy format)
    """
    from pathlib import Path

    file_list = Path(root_dir).rglob('*featuresN.hdf5')
    file_list = [str(file) for file in file_list]

    filenames = pd.DataFrame(file_list, columns=['file_name'])
    filenames.insert(0, 'file_id', np.arange(len(file_list)))

    return filenames

#%% testing
if __name__ == "__main__":

    root_dir = ("/Volumes/behavgenom$/Adam/Screening/"
                + "Syngenta_multi_dose/18-11-13-SYN/Results")

    filenames = get_filenames(root_dir)
    filenames = select_filenames(filenames, None, ['Set2'])
    feat_file = ("/Volumes/behavgenom$/Adam/Screening/Syngenta_multi_dose/"
                 + "18-11-13-SYN/Results/18-11-13_SYN_AMR_1/Set2/"
                 + "Set2_Ch2_13112018_192908_featuresN.hdf5")
    # series = read_timeseries(feat_file)
