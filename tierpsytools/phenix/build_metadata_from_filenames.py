#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:34:22 2019

@author: em812
"""

from pathlib import Path
import pandas as pd
    
def get_digit(string,dtype=float):
    if dtype == int:
        fun = int
    else:
        fun = float
        
    digit = [s for s in string if s.isdigit()]
    digit = ''.join(digit)
    digit = fun(digit)
    
    return digit

def build_meta_cendr(root_results,glob_keyword,recursive=False):
    if recursive:
        result_files = [file for file in Path(root_results).rglob(glob_keyword)]
    else:
        result_files = [file for file in Path(root_results).glob(glob_keyword)]
    
    metadata = []
    for file in result_files:
        strain = file.stem.split('_')[0]
        n_worms = get_digit(file.stem.split('_')[1],dtype=int)
        set_n = get_digit(file.stem.split('_')[3],dtype=int)
        pos = get_digit(file.stem.split('_')[4],dtype=int)
        camera = get_digit(file.stem.split('_')[5],dtype=int)
        date = int(file.stem.split('_')[6])
        
        tmp_meta = pd.DataFrame([[strain,n_worms,set_n,pos,camera,date,str(file)]],columns=['strain','n_worms','set','pos','camera','date','file_name'])
        metadata.append(tmp_meta)
        
    metadata = pd.concat(metadata,axis=0,sort=False)
    metadata.reset_index(drop=True,inplace=True)
        
    return metadata
    
def build_meta_singleworms(root_results,glob_keyword,keep_all_keywords=None,keep_any_keywords=None,recursive=False):
    import numpy as np
    
    if recursive:
        result_files = [file for file in Path(root_results).rglob(glob_keyword)]
    else:
        result_files = [file for file in Path(root_results).glob(glob_keyword)]
    
    metadata = []
    for file in result_files:
        if keep_all_keywords is not None:
            if not np.all([key in str(file) for key in keep_all_keywords]):
                continue
        if keep_any_keywords is not None:
            if not np.any([key in str(file) for key in keep_any_keywords]):
                continue
        strain = file.parts[-6]
        ventral_direction = file.parts[-2]
        if file.stem.find('R_')>=0:
            year = file.stem[file.stem.upper().find('R_')+2:file.stem.upper().find('R_')+6]
        elif file.stem.find('L_')>=0:
            year = file.stem[file.stem.upper().find('L_')+2:file.stem.upper().find('L_')+6]
        else:
            year = file.stem[file.stem.find('_20')+1:file.stem.find('_20')+5]
        tmp_meta = pd.DataFrame([[strain,ventral_direction,int(year),str(file)]],columns=['strain','ventral_direction','year','file_name'])
        metadata.append(tmp_meta)
        
    metadata = pd.concat(metadata,axis=0,sort=False)
    metadata.reset_index(drop=True,inplace=True)
    
    return metadata
    
def build_meta_singleworm_quiescence(root_results,glob_keyword,recursive=False):
    if recursive:
        result_files = [file for file in Path(root_results).rglob(glob_keyword)]
    else:
        result_files = [file for file in Path(root_results).glob(glob_keyword)]
        
    metadata = []
    for file in result_files:
        strain = file.parts[-2]
        chanel = file.stem.split('_')[1]
        date = file.stem.split('_')[2]
        
        tmp_meta = pd.DataFrame([[strain,chanel,date,str(file)]],columns=['strain','chanel','date','filename'])
        metadata.append(tmp_meta)
        
    metadata = pd.concat(metadata,axis=0,sort=False)
    metadata.reset_index(drop=True,inplace=True)
        
    return metadata


def match_metadata_and_clean_features(features,filenames,metadata,feat_meta_cols=['file_id']):
    
    ## The features dataframe is expected to have all the feat_meta_cols columns
    for ft in feat_meta_cols:
        if ft not in features.columns:
            raise KeyError('The feature dataframe does not have a \'{}\' column.'.format(ft))
    
    ## The filenames dataframe is expected to have a 'file_id' column and a 'file_name' column
    for ft in ['file_id','file_name']:
        if ft not in filenames.columns:
            raise KeyError('The filenames dataframe does not have a \'{}\' column.'.format(ft))
            
    ## The metadata dataframe is expected to have a 'file_name' column
    if 'file_name' not in metadata.columns:
        raise KeyError('The metadata dataframe does not have a \'file_name\' column.')
        
    # Match all metadata to features dataframe
    feat_metadata = features.loc[:,feat_meta_cols].copy()
    
    feat_metadata.insert(0,'file_name',feat_metadata['file_id'].map(dict(filenames[['file_id','file_name']].values)))
    
    for dt in metadata.columns.difference(feat_metadata.columns):
        feat_metadata.insert(0,dt,feat_metadata['file_name'].map(dict(metadata[['file_name',dt]].values)))
    
    # Clean features    
    features = features[features.columns.difference(feat_meta_cols)]
    
    return feat_metadata,features