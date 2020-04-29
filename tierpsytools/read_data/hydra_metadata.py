#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:52:14 2020

@author: em812
"""

import pandas as pd
import pdb

def add_bluelight_label(
        meta, filename_column='imgstore_name', split_string='_', location=3
        ):
    from pathlib import Path

    bluelight = meta[filename_column].apply(
        lambda x: Path(x).stem.split(split_string)[location])
    meta.insert(0, 'bluelight', bluelight)
    return meta


def imgstore_name_from_filename(filename, path_levels=[-3,-1]):
    """


    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    path_levels : TYPE, optional
        DESCRIPTION. The default is [-3,-1].

    Returns
    -------
    imgstore_name : TYPE
        DESCRIPTION.

    """
    from pathlib import Path

    imgstore_name = '/'.join(Path(filename).parts[path_levels[0]:path_levels[1]])
    return imgstore_name


def read_hydra_metadata(
        feat, fname, meta,
        feat_id_cols = ['file_id', 'well_name', 'is_good_well'],
        add_bluelight=True, bluelight_label_location_in_imgstore_stem=3):
    # TODO: change the default position counting from the end or look for specific words.
    """
    Creates matching features and metadata dfs from the .csv files of a hydra
    screening (assuming the standardized format of tierpsy and hydra metadata
    from tierpsytools).

    Parameters
    ----------
    feat : TYPE
        DESCRIPTION.
    fname : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    feat_id_cols : TYPE, optional
        DESCRIPTION. The default is ['file_id', 'well_name', 'is_good_well'].
    add_bluelight : TYPE, optional
        DESCRIPTION. The default is True.
    bluelight_label_location_in_imgstore_stem : TYPE, optional
        DESCRIPTION. The default is 3.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    newmeta : TYPE
        DESCRIPTION.

    """
    if 'filename' in fname:
        filename = 'filename'
    elif 'file_name' in fname:
        filename = 'file_name'
    else:
        raise ValueError('The filenames dataframe needs to have a filename column.')

    fname['imgstore_name'] = fname[filename].apply(
        lambda x: imgstore_name_from_filename(x,path_levels=[-3,-1]))

    newmeta = feat[feat_id_cols]

    newmeta.insert(
        0, 'imgstore_name', newmeta['file_id'].map(
            dict(fname[['file_id', 'imgstore_name']].values))
        )

    newmeta = pd.merge(
        newmeta, meta, on=['imgstore_name','well_name'], how='left'
        )

    if add_bluelight:
        newmeta = add_bluelight_label(
            newmeta, location=bluelight_label_location_in_imgstore_stem)

    assert newmeta.shape[0] == feat.shape[0]

    return feat[feat.columns.difference(feat_id_cols)], newmeta

def align_bluelight_conditions(
        feat, meta,
        how = 'outer',
        return_separate_feat_dfs = False,
        bluelight_specific_meta_cols = ['imgstore_name',
                                        'file_id', 'bluelight'],
        merge_on_cols = ['date_yyyymmdd','imaging_plate_id','well_name']
        ):
    """
    Concatenates the features from the three bluelight conditions for each well
    and creates matching features and metadata dataframes for the concatenated
    feature matrix.

    !: is bad well not considered bluelight specific!

    Parameters
    ----------
    feat : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    how : TYPE, optional
        DESCRIPTION. The default is 'outer'.
    return_separate_feat_dfs : TYPE, optional
        DESCRIPTION. The default is False.
    bluelight_specific_meta_cols : TYPE, optional
        DESCRIPTION. The default is ['imgstore_name', 'file_id', 'bluelight'].
    merge_on_cols : TYPE, optional
        DESCRIPTION. The default is ['date_yyyymmdd','imaging_plate_id','well_name'].

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    bluelight_conditions = ['prestim', 'bluelight', 'poststim']

    # The metadata columns that are the same for all bluelight conditions
    shared_meta_cols = meta.columns.difference(
        bluelight_specific_meta_cols+merge_on_cols).to_list()

    # Store the feature names
    feat_names = feat.columns.to_list()

    # Concatenate the metadata with the features for a uniform merge based on
    # metadata
    feat = pd.concat([meta, feat], axis=1)

    # Merge based on the merge_on_cols columns
    feat = feat.set_index(merge_on_cols)

    feat = feat[feat['bluelight']=='prestim'].join(
        feat[feat['bluelight']=='bluelight'],
        how=how, lsuffix='', rsuffix='_bluelight').join(
        feat[feat['bluelight']=='poststim'],
        how=how, lsuffix='', rsuffix='_poststim')

    # Add the prestim suffix to the bluelight-specific columns
    feat = feat.rename(
            columns={ft:ft+'_prestim' for ft in bluelight_specific_meta_cols+feat_names})

    # Derive the feature column names in the merged dataframe and use them to
    # split again the features from the metadata
    feat_cols = ['_'.join([ft,blue])
                 for blue in bluelight_conditions
                 for ft in feat_names]

    feat.reset_index(drop=False, inplace=True)
    meta = feat[feat.columns.difference(feat_cols)]
    feat = feat[feat_cols]

    # The shared meta columns might have nan values in some of the conditions
    # because of the outer merge. I want to keep only one of these columns for
    # all the conditions. So I will replace the nans in the prestim version
    # with the non-nan values from the other conditions and then drop the
    # columns of the other conditions.
    for col in shared_meta_cols:
        for blue in bluelight_conditions[1:]:
            meta.loc[meta[col].isna(), col] = \
                meta.loc[meta[col].isna(), '_'.join([col, blue])]
            meta = meta.drop(columns=['_'.join([col, blue])])

    # Return
    if return_separate_feat_dfs:
        prestim = [ft for ft in feat.columns if '_prestim' in ft]
        bluelight = [ft for ft in feat.columns if '_bluelight' in ft]
        poststim = [ft for ft in feat.columns if '_poststim' in ft]
        return (feat[prestim], feat[bluelight], feat[poststim]), meta
    else:
        return feat, meta

