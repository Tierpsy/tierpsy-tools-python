#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:01:19 2020

@author: em812
"""

from tierpsytools import AUX_FILES_DIR, EXAMPLES_DIR
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances_argmin_min
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf

#%% Input data
n_clusters = 100
method='complete'
metric='correlation'

root_in = Path(EXAMPLES_DIR) / 'fingerprints' / 'sample_data'

feat_file = root_in/ 'features_align_blue=False.csv'
meta_file = root_in/ 'metadata_align_blue=False.csv'

feat = pd.read_csv(feat_file)
meta = pd.read_csv(meta_file)

#%% Cluster features
feat = impute_nan_inf(feat)

column_linkage = linkage(feat.T, method=method, metric=metric)

clusters = fcluster(column_linkage, n_clusters, criterion='maxclust')

un,n = np.unique(clusters, return_counts=True)

#print(n)

# Get cluster centers
cluster_centers = (feat.T).groupby(by=clusters).mean()
# get the index of the feature closest to the centroid of the cluster
central, _ = pairwise_distances_argmin_min(cluster_centers, feat.T, metric='cosine')
assert(np.unique(central).shape[0]==n_clusters)

# get the feature name of the feature closest to the centroid of the cluster
central = feat.columns.to_numpy()[central]

#%% Make dataframe
df = pd.DataFrame(index=feat.columns, columns=['group_label', 'stat_label', 'motion_label'])
df['group_label'] = clusters

stats = np.array(['10th', '50th', '90th', 'IQR'])
df['stat_label'] = [ np.unique([x for x in stats if x in ft]) for ft in df.index]
df['stat_label'] = df['stat_label'].apply(lambda x: x[0] if len(x)==1 else np.nan)
motions = np.array(['forward', 'backward', 'paused'])
df['motion_label'] = [ [x for x in motions if x in ft] for ft in df.index]
df['motion_label'] = df['motion_label'].apply(lambda x: x[0] if len(x)==1 else np.nan)

df['representative_feature'] = False
df.loc[central, 'representative_feature'] = True

df = df.fillna('none')
df.to_csv('feature_clusters.csv')
