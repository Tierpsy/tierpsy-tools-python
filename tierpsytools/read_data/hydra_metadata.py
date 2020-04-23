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
    from pathlib import Path

    imgstore_name = '/'.join(Path(filename).parts[path_levels[0]:path_levels[1]])
    return imgstore_name


def read_hydra_metadata(
        feat, fname, meta,
        feat_id_cols = ['file_id', 'well_name', 'is_good_well'],
        add_bluelight=True, bluelight_label_location_in_imgstore_stem=3):

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
        DESCRIPTION. The default is ['imgstore_name',                                        'file_id', 'bluelight'].
    merge_on_cols : TYPE, optional
        DESCRIPTION. The default is ['date_yyyymmdd','imaging_plate_id','well_name'].

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    bluelight_conditions = ['prestim', 'bluelight', 'poststim']

    distinct_meta_cols = bluelight_specific_meta_cols
    shared_meta_cols = meta.columns.difference(distinct_meta_cols+merge_on_cols).to_list()

    feat_names = feat.columns.to_list()

    feat = pd.concat([meta, feat], axis=1)

    feat = feat.set_index(merge_on_cols)

    feat = feat[feat['bluelight']=='prestim'].join(
        feat[feat['bluelight']=='bluelight'],
        how=how, lsuffix='', rsuffix='_bluelight').join(
        feat[feat['bluelight']=='poststim'],
        how=how, lsuffix='', rsuffix='_poststim')

    feat = feat.rename(
            columns={ft:ft+'_prestim' for ft in distinct_meta_cols+feat_names})

    feat_cols = ['_'.join([ft,blue]) for blue in bluelight_conditions for ft in feat_names]

    feat.reset_index(drop=False, inplace=True)
    meta = feat[feat.columns.difference(feat_cols)]
    feat = feat[feat_cols]

    for col in shared_meta_cols:
        for blue in bluelight_conditions[1:]:
            meta.loc[meta[col].isna(), col] = meta.loc[meta[col].isna(), '_'.join([col, blue])]
            meta = meta.drop(columns=['_'.join([col, blue])])

    if return_separate_feat_dfs:
        return (feat[[ft for ft in feat.columns if '_prestim' in ft]], \
            feat[[ft for ft in feat.columns if '_bluelight' in ft]], \
            feat[[ft for ft in feat.columns if '_poststim' in ft]]), meta
    else:
        return feat, meta