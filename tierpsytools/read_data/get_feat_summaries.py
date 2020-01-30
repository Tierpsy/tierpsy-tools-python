#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:16:50 2019

@author: em812
"""
import pandas as pd
import numpy as np
from time import time

def read_tierpsy_feat_summaries(feat_file,filenames_file,drop_ventral=True):
    """
    Read the existing feature summaries and filenames summaries files produced
    by tierpsy into dataframes.
    (Very basic wrapper to save a few lines in the analysis script)
    Input:
        feat_file = path to feature summaries file
        filenames_file = path to filenames summaries file
        drop_ventral = if True the ventrally signed features are dropped
    """
    from tierpsytools.feature_processing.filter_features import drop_ventrally_signed
    
    filenames = pd.read_csv(filenames_file)
    features = pd.read_csv(feat_file,index_col=1)
    
    if drop_ventral:
        features = drop_ventrally_signed(features)
    
    return filenames,features

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