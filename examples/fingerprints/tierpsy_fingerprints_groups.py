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

    #%% Read
    feat = pd.read_csv(feat_file)
    meta = pd.read_csv(meta_file)

    # feat = feat[[ft for ft in feat.columns if '_IQR' not in ft]]
    saveto = Path().cwd() / 'results_groups'
    saveto.mkdir(exist_ok=True)

    #%% Get fingerprints
    fingers = {}
    for strain in meta.worm_strain.unique()[:3]:
        if strain == 'N2':
            continue
        print('Getting fingerprint of strain {}....'.format(strain))
        (saveto/strain).mkdir(exist_ok=True)

        mask = meta['worm_strain'].isin(['N2', strain])

        finger = tierpsy_fingerprints(
            bluelight=True, test='Mann-Whitney', multitest_method='fdr_by',
            significance_threshold=0.05, groups=None, test_results=None)

        # Fit the fingerprint (run univariate tests and create the profile)
        finger.fit(feat[mask], meta.loc[mask, 'worm_strain'], control='N2')

        # Plot and save the fingerprint for this strain
        finger.plot_fingerprints(merge_bluelight=False,
                                saveto=saveto/strain/'fingerprint.png')

        # Plot and save boxplots for all the representative features
        finger.plot_boxplots(feat[mask], meta.loc[mask, 'worm_strain'], saveto/strain, control='N2')

        # Store the fingerprint object
        fingers[strain] = finger

    #%% Plot fingerprints
    fig, ax = plt.subplots(len(fingers), 1, figsize=(20,5*len(fingers)))
    for i, (strain,finger) in enumerate(fingers.items()):
        finger.plot_fingerprints(
            merge_bluelight=True, ax=ax[i], title=strain, plot_colorbar=False)

    fig.subplots_adjust(bottom=0.06)
    cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.008])
    c = plt.colorbar(sm, orientation='horizontal', cax=cbar_ax) #,     fig.colorbar(im, cax=cbar_ax)
    c.set_label('ratio of significant features') #, labelpad=-40, y=1.15, rotation=0)

    #
    plt.savefig(saveto/'effect_medians.pdf')

