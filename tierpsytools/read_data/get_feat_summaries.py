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


def create_featsdtypesdict(fname):
    featnames = pd.read_csv(fname, comment='#', nrows=0).columns.to_list()
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

    if asfloat32:
        dtypes_dict = create_featsdtypesdict(feat_file)
    else:
        dtypes_dict = None

    filenames = pd.read_csv(filenames_file, comment='#')
    features = pd.read_csv(feat_file, index_col=1,
                           comment='#', dtype=dtypes_dict)


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

    filenames = pd.DataFrame(file_list, columns=['filename'])
    filenames.insert(0, 'file_id', np.arange(len(file_list)))
    filenames['is_good'] = True

    return filenames

def read_feat_stats(filename, log_dir=None):
    """
    Read the feature stats from a *featuresN.hdf5 file from tierpsy.
    """
    from time import time
    from pathlib import Path

    try:
        feat = pd.read_hdf(filename, key='features_stats')
    except:
        if log_dir is not None:
            with open(Path(log_dir)/'error_{:6.6f}.err'.format(time()), 'w') as fid:
                fid.write('Error: file empty or there is no features_stats dataframe in file:\n')
                fid.write(filename)
        return pd.DataFrame([]), None

    if feat.empty:
        return feat, None
    else:
        if 'well_name' in feat:
            well_feat = []
            for well in feat['well_name'].unique():
                tmp_feat = feat.loc[feat['well_name']==well, ['name','value']]
                tmp_feat = pd.DataFrame(
                    tmp_feat['value'].values.reshape(1,-1),
                    columns=tmp_feat['name'].values
                    )
                tmp_feat.insert(0, 'well_name', [well])
                well_feat.append(tmp_feat)
            feat = pd.concat(well_feat, sort=False)
            is_split_fov = True
        else:
            feat = pd.DataFrame(
                feat['value'].values.reshape(1,-1),
                columns=feat['name'].values
                )
            is_split_fov = False
    return feat, is_split_fov

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
    for ifl,(fileid,file) in enumerate(filenames[['file_id','filename']].values):
        file_time = time()
        print('Reading features stats from file {} of {}'.format(ifl+1,filenames.shape[0]))
        ft, is_split_fov = read_feat_stats(file)
        if ft.empty:
            filenames[filenames['file_id']==fileid, 'is_good'] = False
            continue
        ft['file_id'] = fileid
        features.append(ft)
        print('File read in {} sec.'.format(time()-file_time))
    print('Done reading in {} sec.'.format(time()-start_time))
    features = pd.concat(features, axis=0, sort=False)

    features.reset_index(drop=True,inplace=True)

    if drop_ventral:
        features = drop_ventrally_signed(features)

    return filenames,features

def drop_ventrally_signed_names(feat_names):
    """
    EM: drops the ventrally signed features
    Param:
        features_names = list of features names
    Return:
        filtered_names = list of features names without ventrally signed
    """

    absft = [ft for ft in feat_names if '_abs' in ft]
    ventr = [ft.replace('_abs', '') for ft in absft]

    filtered_names = list(set(feat_names).difference(set(ventr)))

    return filtered_names

def select_and_sort_columns(df, selected_feat, id_cols):
    """
    Sorts the columns of the feat summaries dataframe to make sure that each
    line written in the features summaries file contains the same features with
    the same order. If a feature has not been calculated in the df, then a nan
    value is added.
    """

    not_existing_cols = [col for col in selected_feat if col not in df.columns]

    if len(not_existing_cols) > 0:
        for col in not_existing_cols:
            df[col] = np.nan

    df = df[[x for x in id_cols if x in df.columns] + selected_feat]

    return df

def remove_files_already_read(filenames, f2):

    finished_files = pd.read_csv(f2, index_col=None, comment='#')

    filenames = filenames[~filenames['filename'].isin(
        finished_files['filename'].to_list()
        )]

    filenames.reset_index(drop=True, inplace=True)
    filenames['file_id'] = filenames.index.to_list() + \
        finished_files['file_id'].max() + 1

    return filenames

def write_all_feat_summaries_to_file(
        root_dir,
        drop_ventral=True,
        save_to=None,
        feat_sum_filename='features_summaries_tierpsy_plate.csv',
        filenames_sum_filename='filenames_summaries_tierpsy_plate.csv',
        append_to_existing=False
        ):
    """
    Same as get_all_feat_summaries BUT grows csv file instead of growing a
    dataframe and keeping it in memory (more efficient when you have many files
    to read).

    Get feature summaries reading the feat_stats from the *_featuresN files
    (instead of using the tierpsy summarizer gui).
    Parameters:
        root_dir: path
            Results root directory
        drop_ventral: boolean, optional
            If True the ventrally signed features are dropped. Default is True
        save_to: path
            Path of directory where to save the feature summaries file and the
            filenames file. If None, then the root_dir is used. Default is None.
        feat_sum_filename: string, optional
            The features summaries filename.
    """
    from tierpsytools.feature_processing.filter_features import drop_ventrally_signed
    from tierpsytools import AUX_FILES_DIR
    from pathlib import Path
    import pdb

    feat_id_cols = ['file_id', 'well_name']

    if save_to is None:
        save_to = root_dir
    Path(save_to).mkdir(exist_ok=True)

    log_dir = Path(save_to)/'feat_summaries_error_log'
    log_dir.mkdir(exist_ok=True)

    # Create full file paths for the feature summaries file and the filenames
    # summaries file
    f1 = Path(save_to) / feat_sum_filename
    f2 = Path(save_to) / filenames_sum_filename

    if append_to_existing and not (f1.exists() and f2.exists()):
        raise ValueError('There are no existing summaries files to append')
    elif not append_to_existing and (f1.exists() and f2.exists()):
        raise ValueError(
            'Summary files with this name already exist at this location.'+
            'Remove or rename the existing files to run this function.')

    # Get all tierpsy features names that we expect from auxiliary files
    if append_to_existing:
        feat_names = pd.read_csv(
            f1, index_col=None, nrows=1, comment='#').columns.difference(
                feat_id_cols, sort=False).to_list()
        with open(f1, 'a') as fid:
            fid.write("\n")
        with open(f2, 'a') as fid:
            fid.write("\n")
    else:
        feat_names = pd.read_csv(
            Path(AUX_FILES_DIR) / 'tierpsy_features_full_names.csv', header=None,
            index_col=None)[0].to_list()
        if drop_ventral:
            feat_names = drop_ventrally_signed_names(feat_names)

        # Write the headers in the features and filenames summaries file
        with open(f1, 'w') as fid:
            fid.write(','.join(feat_id_cols + feat_names)+"\n")
        with open(f2, 'w') as fid:
            fid.write(','.join(['file_id', 'filename', 'is_good'])+"\n")

    # Create the filenames summaries dataframe
    filenames = get_filenames(root_dir)

    if append_to_existing:
        filenames = remove_files_already_read(filenames, f2)

    # pdb.set_trace()

    start_time=time()
    for ifl, (fileid, file) in enumerate(filenames[['file_id','filename']].values):
        file_time = time()
        print('Reading features stats from file {} of {}'.format(ifl+1,filenames.shape[0]))
        ft, is_split_fov = read_feat_stats(file, log_dir=log_dir)

        if ft.empty:
            filenames.loc[filenames['file_id']==fileid, 'is_good'] = False
        else:
            # Write the feature summaries to file (only if is_good)
            ft['file_id'] = fileid
            ft = select_and_sort_columns(ft, feat_names, feat_id_cols)

            with open(f1,'a') as fid:
                ft.to_csv(fid, header=False, index=False)

        # Write the filenames summaries line to file
        with open(f2,'a') as fid:
            filenames[filenames['file_id']==fileid].to_csv(
                fid, header=False, index=False)

        print('File read in {} sec.'.format(time()-file_time))

    print('Done reading in {} sec.'.format(time()-start_time))

    return

if __name__=="__main__":

    # root_dir = '/Volumes/behavgenom$/Ida/Data/Hydra/SyngentaStrainScreen/Results/'
    # write_all_feat_summaries_to_file(root_dir,drop_ventral=True)

    # filename = '/Volumes/behavgenom$/Ida/Data/Hydra/SyngentaStrainScreen/Results/20200129/run1_syngenta_bluelight_20200129_141526.22956831/metadata_skeletons.hdf5'
    filename = '/Volumes/behavgenom$/Ida/Data/Hydra/SyngentaStrainScreen/Results/20200129/run1_syngenta_bluelight_20200129_141526.22956831/metadata_featuresN.hdf5'
    feat, is_split = read_feat_stats(filename, log_dir='.')
