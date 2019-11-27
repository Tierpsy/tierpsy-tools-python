#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: em812

"""
from sklearn.feature_selection import f_classif,SelectKBest,RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import test_train_split
from sklearn.preprocessing import scale
import pandas as pd

# Paths to input files
feature_file = '' # file containing feature matrix
metadata_file = '' # file containing group/class info for each sample (row) in the feature matrix

# How many features to select?
nfeat = 10

features = pd.read_csv(feature_file)
metadata = pd.read_csv(metadata_file)

feat_cv,feat_cv,meta_train,meta_test = test_train_split()

scaled_feat = scale(feat_cv)
y = meta_train['group'].values

# Method 1: select k best
skb = SelectKBest(core_func=f_classif,k=nfeat)
skb.fit(scaled_feat,y)

skb_selected_features = skb.transform(features)
skb_selected_features = pd.DataFrame(skb_selected_features,columns=features.columns,index=features.index)

# check the correlation between the selected features?
skb_corr_matrix = skb_selected_features.corr()

# Method 2: RFE
estimator = LogisticRegression()
rfe = RFE(estimator,n_features_to_select=nfeat,step=10)
rfe.fit(scaled_feat,y)

rfe_selected_features = rfe.transform(features)
rfe_selected_features = pd.DataFrame(rfe_selected_features,columns=features.columns,index=features.index)

# check the correlation between the selected features (should be lower than method 1)?
rfe_corr_matrix = skb_selected_features.corr()

# From here on you can do your analysis with the selected features (goal achieved)