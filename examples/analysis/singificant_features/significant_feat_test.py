#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:32:50 2020

@author: em812
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tierpsytools.analysis.significant_features import mRMR_feature_selection, k_significant_feat
from sklearn.model_selection import cross_val_score

#%% Input for aligned bluelight
data_file = 'sample_data.csv'

estimator = Pipeline([('scaler', StandardScaler()),
                      ('estimator', LogisticRegression())
                      ])

#%%
data = pd.read_csv(data_file, index_col=None)

y = data['worm_strain'].values
data = data.drop(columns='worm_strain')

#%%
mrmr_feat_set, mrmr_scores, mrmr_support = mRMR_feature_selection(
        data, k=10, y_class=y,
        redundancy_func='pearson_corr', relevance_func='kruskal',
        n_bins=4, mrmr_criterion='MIQ',
        plot=True, k_to_plot=5, close_after_plotting=False,
        saveto=None, figsize=None
        )

cv_scores_mrmr = cross_val_score(estimator, data[mrmr_feat_set], y, cv=5)
print('MRMR')
print(np.mean(cv_scores_mrmr))

#%%
feat_set, scores, support = k_significant_feat(
        data, y, k=10, score_func='f_classif', scale=None,
        plot=True, k_to_plot=5, close_after_plotting=False,
        saveto=None, figsize=None, title=None, xlabel=None
        )

cv_scores = cross_val_score(estimator, data[feat_set], y, cv=5)
print('k-best')
print(np.mean(cv_scores))
