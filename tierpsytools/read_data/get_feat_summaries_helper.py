#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:16:01 2021

@author: em812
"""
import pandas as pd
import numpy as np

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

    from tierpsytools.preprocessing.filter_data import (
        drop_ventrally_signed)

    if asfloat32:
        dtypes_dict = create_featsdtypesdict(feat_file)
    else:
        dtypes_dict = None

    filenames = pd.read_csv(filenames_file, comment='#')
    features = pd.read_csv(feat_file, comment='#', dtype=dtypes_dict)


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
