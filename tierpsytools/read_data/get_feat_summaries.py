#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:16:50 2019

@author: em812
"""
import pandas as pd
import numpy as np
from time import time


def count_metadata_lines_in_csv(fname):
    """count_metadata_lines_in_csv
    Return the number of lines starting with # in the input file.
    e.g. if there are 3 lines starting with #, return 3.
    Then line 3 is the first non-metadata line to read (because lines
    are indexed from 0)"""
    n_metadata_lines = 0
    with open(fname, 'r') as fid:
        for line in fid:
            if line.startswith('#'):
                n_metadata_lines += 1
            else:
                return n_metadata_lines


def create_featsdtypesdict(fname, n_metadata_lines=None):
    if n_metadata_lines is None:
        n_metadata_lines = count_metadata_lines_in_csv(fname)
    with open(fname, 'r') as fid:
        for i, line in enumerate(fid):
            if i == n_metadata_lines:
                featnames = line.strip('\n').split(',')
                break
    dtypes_dict = {}
    dtypes_dict['file_id'] = np.int64
    if 'well_name' in featnames:
        dtypes_dict['well_name'] = str
    if 'is_good_well' in featnames:
        dtypes_dict['is_good_well'] = bool
    for feat in featnames:
        if feat not in ['file_id', 'well_name', 'is_good_well']:
            dtypes_dict[feat] = np.float32
    return dtypes_dict


def read_tierpsy_feat_summaries(
        feat_file, filenames_file, drop_ventral=True, asfloat32=False):
    """
    Read the existing feature summaries and filenames summaries files produced
    by tierpsy into dataframes.
    (Very basic wrapper to save a few lines in the analysis script)
    Input:
        feat_file = path to feature summaries file
        filenames_file = path to filenames summaries file
        drop_ventral = if True the ventrally signed features are dropped
    """
    from tierpsytools.feature_processing.filter_features import (
        drop_ventrally_signed)

    headerline_features = count_metadata_lines_in_csv(feat_file)
    if asfloat32:
        dtypes_dict = create_featsdtypesdict(
            feat_file,
            n_metadata_lines=headerline_features)
    else:
        dtypes_dict = None

    features = pd.read_csv(feat_file,
                           skiprows=headerline_features,
                           header=0,
                           dtype=dtypes_dict)

    headerline_filenames = count_metadata_lines_in_csv(filenames_file)
    filenames = pd.read_csv(filenames_file,
                            skiprows=headerline_filenames,
                            header=0)

    if drop_ventral:
        features = drop_ventrally_signed(features)

    return filenames, features


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

    filenames = pd.DataFrame(file_list,columns=['file_name'])
    filenames.insert(0,'file_id',np.arange(len(file_list)))

    return filenames

def read_feat_stats(filename):
    """
    Read the feature stats from a *featuresN.hdf5 file from tierpsy.
    """
    import h5py

    with h5py.File(filename,'r') as f:
        if pd.DataFrame(f['features_stats']['value']).empty:
            feat = pd.DataFrame([],index=[0])
        else:
            feat = pd.DataFrame([],index=[0])
            name = f['features_stats']['name']
            value = f['features_stats']['value']
            for nm,vl in zip(name,value):
                nm = nm.decode()
                feat.loc[0,nm] = vl
    return feat

def get_all_feat_summaries(root_dir,drop_ventral=True):
    """
    Get feature summaries reading the feat_stats from the *_featuresN files
    (instead of using the tierpsy summarizer gui).
    Input:
        root_dir = results root directory
        drop_ventral = if True the ventrally signed features are dropped
    Return:
        filenames = dataframe with filenames summaries (tierpsy format)
        features = dataframe with features summaries (tierpsy format)
    """
    from tierpsytools.feature_processing.filter_features import drop_ventrally_signed

    filenames = get_filenames(root_dir)

    features = []
    start_time=time()
    for ifl,(fileid,file) in enumerate(filenames[['file_id','file_name']].values):
        file_time = time()
        print('Reading features stats from file {} of {}'.format(ifl+1,filenames.shape[0]))
        ft = read_feat_stats(file)
        ft['file_id'] = fileid
        features.append(ft)
        print('File read in {} sec.'.format(time()-file_time))
    print('Done reading in {} sec.'.format(time()-start_time))
    features = pd.concat(features,axis=0,sort=False)

    features.reset_index(drop=True,inplace=True)

    if drop_ventral:
        features = drop_ventrally_signed(features)

    return filenames,features