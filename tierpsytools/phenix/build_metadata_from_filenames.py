#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:34:22 2019

@author: em812
"""

from pathlib import Path
import pandas as pd
import re
import datetime

def get_digit(string, dtype=float):
    if dtype == int:
        fun = int
    else:
        fun = float

    digit = [s for s in string if s.isdigit()]
    digit = ''.join(digit)
    digit = fun(digit)

    return digit

def meta_cendr_from_filenames(filenames):

    filenames = [Path(file) for file in filenames]

    metadata = []
    for file in filenames:
        strain = file.stem.split('_')[0]
        n_worms = get_digit(file.stem.split('_')[1],dtype=int)
        set_n = get_digit(file.stem.split('_')[3],dtype=int)
        pos = get_digit(file.stem.split('_')[4],dtype=int)
        camera = get_digit(file.stem.split('_')[5],dtype=int)
        date = int(file.stem.split('_')[6])

        tmp_meta = pd.DataFrame([[strain,n_worms,set_n,pos,camera,date,str(file)]],columns=['strain','n_worms','set','pos','camera','date','filename'])
        metadata.append(tmp_meta)

    metadata = pd.concat(metadata,axis=0,sort=False)
    metadata.reset_index(drop=True,inplace=True)

    return metadata

def build_meta_cendr(root_results, glob_keyword, recursive=False):
    if recursive:
        result_files = [file for file in Path(root_results).rglob(glob_keyword)]
    else:
        result_files = [file for file in Path(root_results).glob(glob_keyword)]

    metadata = meta_cendr_from_filenames(result_files)

    return metadata

def meta_singleworms_from_filenames(
        filenames, keep_all_keywords=None, keep_any_keywords=None
        ):

    filenames = [Path(file) for file in filenames]

    metadata = []
    for file in filenames:
        if keep_all_keywords is not None:
            if not all([key in str(file) for key in keep_all_keywords]):
                continue
        if keep_any_keywords is not None:
            if not any([key in str(file) for key in keep_any_keywords]):
                continue
        strain = file.parts[-6]
        ventral_direction = file.parts[-2]
        if file.stem.find('R_')>=0:
            year = file.stem[file.stem.upper().find('R_')+2:file.stem.upper().find('R_')+6]
        elif file.stem.find('L_')>=0:
            year = file.stem[file.stem.upper().find('L_')+2:file.stem.upper().find('L_')+6]
        else:
            year = file.stem[file.stem.find('_20')+1:file.stem.find('_20')+5]
        tmp_meta = pd.DataFrame([[strain,ventral_direction,int(year),str(file)]],columns=['strain','ventral_direction','year','filename'])
        metadata.append(tmp_meta)

    metadata = pd.concat(metadata,axis=0,sort=False)
    metadata.reset_index(drop=True,inplace=True)

    return metadata

def build_meta_singleworms(
        root_results, glob_keyword,
        keep_all_keywords=None, keep_any_keywords=None,
        recursive=False
        ):

    if recursive:
        result_files = [file for file in Path(root_results).rglob(glob_keyword)]
    else:
        result_files = [file for file in Path(root_results).glob(glob_keyword)]

    metadata = meta_singleworms_from_filenames(
        result_files, keep_all_keywords=keep_all_keywords,
        keep_any_keywords=keep_any_keywords)

    return metadata

def meta_singleworm_quiescence_from_filenames(filenames):

    filenames = [Path(file) for file in filenames]

    metadata = []
    for file in filenames:
        strain = file.parts[-2]
        chanel = file.stem.split('_')[1]
        date = file.stem.split('_')[2]

        tmp_meta = pd.DataFrame([[strain,chanel,date,str(file)]],columns=['strain','chanel','date','filename'])
        metadata.append(tmp_meta)

    metadata = pd.concat(metadata,axis=0,sort=False)
    metadata.reset_index(drop=True,inplace=True)

    return metadata

def build_meta_singleworm_quiescence(root_results,glob_keyword,recursive=False):
    if recursive:
        result_files = [file for file in Path(root_results).rglob(glob_keyword)]
    else:
        result_files = [file for file in Path(root_results).glob(glob_keyword)]

    metadata = meta_singleworm_quiescence_from_filenames(result_files)

    return metadata


def meta_syngenta_archive_from_filenames(filenames):
    import pdb

    filenames = [str(file).replace('No_Compound','NoCompound') for file in filenames]

    filenames = [Path(file) for file in filenames]

    metadata = []
    for file in filenames:
        tmp_meta = dict()
        tmp_meta['strain'] = [file.stem.split('_')[0]]
        tmp_meta['nworms'] = [get_digit(file.stem.split('_')[1], dtype=int)]
        tmp_meta['drug_name'] = [file.stem.split('_')[2]]
        try:
            tmp_meta['drug_dose'] = [float(file.stem.split('_')[3])]
        except:
            pdb.set_trace()
        tmp_meta['set'] = [get_digit(file.stem.split('_')[4], dtype=int)]
        tmp_meta['position'] = [get_digit(file.stem.split('_')[5], dtype=int)]
        tmp_meta['channel'] = [get_digit(file.stem.split('_')[6], dtype=int)]
        tmp_meta['date'] = [file.stem.split('_')[7]]
        tmp_meta['filename'] = [str(file)]

        tmp_meta = pd.DataFrame(tmp_meta)
        metadata.append(tmp_meta)

    metadata = pd.concat(metadata,axis=0,sort=False)
    metadata.reset_index(drop=True,inplace=True)

    return metadata

def meta_from_filenames_venoms(filelist,
                               extract_channels=True):
    """
    author @ibarlow

    Function that extracts the filenames from the venoms data
    .mjpg vidoes used to acquire the data and 
    and so channels have not been converted to 1,2,3,4,5,6 when transferred to
    behavgenom
    
    Parameters
    ----------
    filelist : list of files to extract metadata from 
        DESCRIPTION.
    extract_channels: Bool
        detault is True

    Returns
    -------
    meta : TYPE
        DESCRIPTION.

    """
    run_regex = r"(?<=run|set)\d{1,}"
    camera_regex =r"(?<=Ch)\d{1,}"
    date_regex = r"(?<=_)\d{8,}(?=_)"  
    channel_dict = {(1, 1): 1,
                    (1, 2): 2,
                    (2, 1): 3,
                    (2, 2): 4,
                    (3, 1): 5,
                    (3, 2): 6}
 
    filenames = [Path(file) for file in filelist]
                           
    meta= []
    for count, file in enumerate(filenames):
        meta.append(pd.Series({'venom_type': file.stem.split('_')[0],
                               'run_number': re.findall(run_regex,
                                                        file.stem)[0],
                               'camera_number': int(re.findall(camera_regex,
                                                           file.stem)[0]),
                               'PC_number': int(file.parts[-3][-1]),
                               'date_YYYYMMDD': datetime.datetime.strptime(
                                   re.search(date_regex,
                                             str(file))[0],
                                   '%d%m%Y').strftime('%Y%m%d'),
                               'filename': file}).to_frame(
                                       ).transpose())
    
    meta = pd.concat(meta)
    meta.reset_index(drop=True, inplace=True)
    if extract_channels:
        meta['channel'] = pd.Series(zip(meta['PC_number'],
                                        meta['camera_number'])).map(
                                            channel_dict)
    else:
        meta.rename(columns = {'camera_number': 'channel'})
        
    meta.sort_values(by=['date_YYYYMMDD',
                        'run_number',
                        'channel'],
                     inplace=True,
                     ignore_index=True)
    meta.reset_index(drop=True,
                     inplace=True)
    
    return meta



def match_metadata_and_clean_features(
        features, filenames, metadata, feat_meta_cols=['file_id']
        ):

    ## The features dataframe is expected to have all the feat_meta_cols columns
    for ft in feat_meta_cols:
        if ft not in features.columns:
            raise KeyError('The feature dataframe does not have a \'{}\' column.'.format(ft))

    ## The filenames dataframe is expected to have a 'file_id' column and a 'filename' column
    for ft in ['file_id','filename']:
        if ft not in filenames.columns:
            raise KeyError('The filenames dataframe does not have a \'{}\' column.'.format(ft))

    ## The metadata dataframe is expected to have a 'filename' column
    if 'filename' not in metadata.columns:
        raise KeyError('The metadata dataframe does not have a \'filename\' column.')

    # Match all metadata to features dataframe
    feat_metadata = features.loc[:,feat_meta_cols].copy()

    feat_metadata.insert(0,'filename',feat_metadata['file_id'].map(dict(filenames[['file_id','filename']].values)))

    for dt in metadata.columns.difference(feat_metadata.columns):
        feat_metadata.insert(0,dt,feat_metadata['filename'].map(dict(metadata[['filename',dt]].values)))

    # Clean features
    features = features[features.columns.difference(feat_meta_cols)]

    return feat_metadata,features