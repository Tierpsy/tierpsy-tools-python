#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:47:04 2021

@author: em812
"""
import pandas as pd
import numpy as np
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools.analysis.fingerprints import tierpsy_fingerprints
from tierpsytools import AUX_FILES_DIR
from pathlib import Path
import matplotlib.pyplot as plt
import pdb
import seaborn as sns
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

#%% TODO: Remove IQR
if __name__=="__main__":

    # Input files
    root_in = Path().cwd() / 'sample_data'
    feat_file = root_in / 'features.csv'
    meta_file = root_in / 'metadata.csv'

    clusters_file = Path().cwd() / 'cluster_features' / 'feature_clusters.csv'

    #%% Read
    feat = pd.read_csv(feat_file)
    meta = pd.read_csv(meta_file)

    # Groups
    clusters = pd.read_csv(clusters_file, index_col=0)

    # feat = feat[[ft for ft in feat.columns if '_IQR' not in ft]]
    saveto = Path().cwd() / 'results_clusters'
    saveto.mkdir(exist_ok=True)

    #%% Get fingerprints
    fingers = {}
    for strain in meta.worm_strain.unique()[:3]:
        if strain == 'N2':
            continue
        print('Getting fingerprint of strain {}...'.format(strain))
        # Make a folder to save the results for this strain
        (saveto/strain).mkdir(exist_ok=True)

        # Create the fingerprint class object
        mask = meta['worm_strain'].isin(['N2', strain])

        finger = tierpsy_fingerprints(
            bluelight=True, test='Mann-Whitney', multitest_method='fdr_by',
            significance_threshold=0.05, groups=clusters, groupby=['group_label'],
            test_results=None, representative_feat='representative_feature')

        # Fit the fingerprint (run univariate tests and create the profile)
        finger.fit(feat[mask], meta.loc[mask, 'worm_strain'], control='N2')

        # Plot and save the fingerprint for this strain
        finger.plot_fingerprints(merge_bluelight=False, feature_names_as_xticks=True,
                                saveto=saveto/strain/'fingerprint.png')

        # Plot and save boxplots for all the representative features
        finger.plot_boxplots(feat[mask], meta.loc[mask, 'worm_strain'], saveto/strain, control='N2')

        # Store the fingerprint object
        fingers[strain] = finger


    #%% Plot fingerprints of differents strain in a comparative plot
    n_fingers = len(fingers.keys())
    fig, ax = plt.subplots(n_fingers, 1, figsize=(20,5*n_fingers))

    for i, (strain,finger) in enumerate(fingers.items()):
        finger.plot_fingerprints(
            merge_bluelight=True, ax=ax[i], title=strain, plot_colorbar=False)

    fig.subplots_adjust(bottom=0.06)
    cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.008])

    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    c = plt.colorbar(sm, orientation='horizontal', cax=cbar_ax) #,     fig.colorbar(im, cax=cbar_ax)
    c.set_label('ratio of significant features') #, labelpad=-40, y=1.15, rotation=0)

    #
    plt.savefig(saveto/'all_fingerprints.png', dpi=500)

