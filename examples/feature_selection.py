#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: em812

"""
from sklearn.feature_selection import f_classif,SelectKBest,RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import pandas as pd

from tierpsytools.read_data.hydra_metadata import read_hydra_metadata

#%% Input

#Paths to input files
# Attention: It is assumed that the features contain no missing values
feature_file = '' # file containing feature matrix
metadata_file = '' # file containing metadata

# How many features to select?
nfeat = 10

#%% Read input files and preprocess
features = pd.read_csv(feature_file, comment='#')
metadata = pd.read_csv(metadata_file, comment='#')

# Test-train split
feat_cv, feat_test, meta_cv, meta_test = train_test_split(features, metadata)
y = meta_cv['date_yyyymmdd'].values

# Scale
feat_cv = feat_cv.loc[:, feat_cv.std()!=0]
feat_test = feat_cv.loc[:, feat_cv.std()!=0]
scaled_feat = scale(feat_cv)
scaled_feat = pd.DataFrame(scaled_feat).fillna(0)

#%% Select features

# Method 1: select k best
skb = SelectKBest(score_func=f_classif, k=nfeat)
skb.fit(scaled_feat,y)

skb_selected_features = features.loc[:, skb.get_support()]

# check the correlation between the selected features?
skb_corr_matrix = skb_selected_features.corr()

# Method 2: RFE
estimator = LogisticRegression()
rfe = RFE(estimator, n_features_to_select=nfeat, step=10)
rfe.fit(scaled_feat, y)

rfe_selected_features = features.loc[:, rfe.get_support()]

# check the correlation between the selected features (should be lower than method 1)?
rfe_corr_matrix = skb_selected_features.corr()

# From here on you can do your analysis with the selected features (goal achieved)