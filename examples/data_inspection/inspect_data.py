#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of data inspection script using synthetic data from 3 hypothetical strains
screened in two different days with bluelight stimulus.

Created on Wed Feb 10 16:52:07 2021

@author: em812
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions
from tierpsytools.preprocessing.filter_data import filter_nan_inf, filter_n_skeletons
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.preprocessing.scaling_class import scalingClass
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#%% Input file paths
# Here, I have a file with the features dataframe and the matching metadata dataframe,
# because the script is based on synthetic data.
# When using tierpsy features, you would normal have three input files:
# the features_summaries file, the filenames_summaries file and the metadata file.
feat_file = Path().cwd() / 'sample_data' / 'features_dataframe.csv'
meta_file = Path().cwd() / 'sample_data' / 'metadata_dataframe.csv'

#%% Read data
# If the input files are the features_summaries, the filenames_summaries and
# metadata files, then you can use this function to make your matching feat
# and meta dataframes:
# feat, meta = read_hydra_metadata(feat_file, fname_file, meta_file)

# For the synthetic data, I just need to read the dataframes:
feat = pd.read_csv(feat_file)
meta = pd.read_csv(meta_file)

# Align the bluelight conditions (one row per well, wide format)
feat, meta = align_bluelight_conditions(
    feat, meta, bluelight_specific_meta_cols=['imgstore_name', 'n_skeletons'],
    merge_on_cols= ['date_yyyymmdd', 'imaging_plate_id', 'well_name'])

#%% Filter data
# Filter rows based on n_skeletons
feat, meta = filter_n_skeletons(
    feat, meta, min_nskel_per_video=2000, min_nskel_sum=None)

# Filter rows based on percentage of nan values
feat = filter_nan_inf(feat, 0.2, 1)
meta = meta.loc[feat.index]

# Filter features based on percentage of nan values
feat = filter_nan_inf(feat, 0.05, 0)

#%% Preprocess data
# Impute the remainig nan values
feat = impute_nan_inf(feat, groupby=None)

# Scale the features (necessary if tou want to do PCA)
scaler = scalingClass(scaling='standardize')
feat = scaler.fit_transform(feat)

#%% Check day-to-day variation
# Get the PCA decomposition
pca = PCA(n_components=2)
Y = pca.fit_transform(feat)

# Plot the samples of each day in the first two PCs
plt.figure()
for day in meta['date_yyyymmdd'].unique():
    plt.scatter(*Y[meta['date_yyyymmdd']==day, :].T, label=day)
plt.legend()

