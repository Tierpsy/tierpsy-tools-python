#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:57:29 2019

@author: em812
"""

from tierpsytools.phenix.build_metadata_from_filenames import build_meta_cendr,match_metadata_and_clean_features
from tierpsytools.phenix.get_feat_summaries import get_all_feat_summaries
from tierpsytools.feature_processing.filter_features import select_feat_set,drop_ventrally_signed

#%%
## Choice to read features from results or from features summaries
# Results
results_root = '/Users/em812/Data/CeNDR/sample_res_files'

# Features summaries
features_sum_file = '/Users/em812/Data/CeNDR/sample_res_files/feature_summaries.csv'

# Output metadata from filenames
meta_file = '/Users/em812/Data/CeNDR/sample_res_files/metadata_from_filenames.csv'

#%%
## Get metadata
metadata = build_meta_cendr(results_root,'*featuresN.hdf5',recursive=False)
metadata.to_csv(meta_file,index=None)

## Get features from featuresN files
# The features dataframe has a column "file_id"
filenames,features = get_all_feat_summaries(results_root)
print(features.shape)

## Match metadata to feature matrix
# Outputs a metadata dataframe with index matching to the feature dataframe index.
# Each row contains all the metadata for the corresponding sample of the feature dataframe.
# When we get this dataframe, we no longer need the "filename" or "metadata" dataframes
# for downstream analysis.
feat_metadata,features = match_metadata_and_clean_features(features,filenames,metadata,feat_meta_cols=['file_id'])

features = drop_ventrally_signed(features)
print(features.shape)

features = select_feat_set(features,'tierpsy_16')
print(features.shape)

