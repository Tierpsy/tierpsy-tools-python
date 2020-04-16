#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:49:21 2020

@author: em812
"""
import pandas as pd
import pdb


def syngenta_from_filenames(feat, fnames):
    from tierpsytools.phenix.build_metadata_from_filenames import meta_syngenta_archive_from_filenames

    meta = meta_syngenta_archive_from_filenames(fnames['file_name'].values)

    newmeta = feat[['file_id']]

    newmeta.insert(0, 'filename',
                   newmeta['file_id'].map(
                       dict(fnames[['file_id', 'file_name']].values)
                       )
                   )

    newmeta.insert(0, 'precipitation',
                   newmeta['file_id'].map(
                       dict(fnames[['file_id', 'precipitation']].values)
                       )
                   )

    newmeta = pd.merge(
        newmeta, meta, on='filename', how='left'
        )

    return feat[feat.columns.difference(['file_id'], sort=False)], newmeta


def syngenta_from_metadata_file(feat, fnames, meta):

    newmeta = feat[['file_id']]

    newmeta.insert(0, 'filename',
                   newmeta['file_id'].map(
                       dict(fnames[['file_id', 'file_name']].values)
                       )
                   )

    newmeta.insert(0, 'precipitation_fnames',
                   newmeta['file_id'].map(
                       dict(fnames[['file_id', 'precipitation']].values)
                       )
                   )

    meta = meta.rename(columns={'results_file_path':'filename'})

    newmeta = pd.merge(
        newmeta, meta, on='filename', how='left'
        )

    return feat[feat.columns.difference(['file_id'], sort=False)], newmeta


def combine_syngenta_dfs(feat1, meta1, feat2, meta2):

    rename_dict = {'N_Worms': 'nworms',
                   'Strain': 'strain',
                   'Camera_N': 'channel',
                   'Rig_Pos': 'position',
                   'Set_N': 'set'}

    meta2 = meta2.rename(columns=rename_dict)

    common_cols = set(meta1.columns.to_list()).intersection(set(meta2.columns.to_list()))

    meta = pd.concat([meta1[common_cols], meta2[common_cols]], axis=0)
    feat = pd.concat([feat1, feat2], axis=0, sort=False)

    feat.reset_index(drop=True)
    meta.reset_index(drop=True)

    return feat, meta
