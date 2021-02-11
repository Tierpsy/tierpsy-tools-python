#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:49:21 2020

@author: em812
"""
import pandas as pd
import pdb


def heterodera_from_metadata_file(feat, fnames, meta):

    meta_cols_in_feat_df = ['file_id', 'n_skeletons']

    newmeta = feat[meta_cols_in_feat_df]

    newmeta.insert(0, 'filename',
                   newmeta['file_id'].map(
                       dict(fnames[['file_id', 'filename']].values)
                       )
                   )

    newmeta.insert(0, 'is_good',
                   newmeta['file_id'].map(
                       dict(fnames[['file_id', 'is_good']].values)
                       )
                   )

    meta = meta.rename(columns={'results_filename': 'filename'})

    newmeta = pd.merge(
        newmeta, meta, on='filename', how='left'
        )

    return feat[feat.columns.difference(meta_cols_in_feat_df, sort=False)], newmeta

