#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:01:07 2020

@author: em812
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

#%% plotting function
def plot_feature_boxplots(
        feat_to_plot, y_class, scores, pvalues=None,
        figsize=None, saveto=None, xlabel=None,
        close_after_plotting=False
        ):

    classes = np.unique(y_class)
    for i,ft in enumerate(feat_to_plot):
        title = ft+'\n'+'score={:.3f}'.format(scores[i])
        if pvalues is not None:
            title+=' - p-value = {}'.format(pvalues[i])
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.boxplot([feat_to_plot.loc[y_class==cl,ft] for cl in classes])
        plt.xticks(list(range(1,classes.shape[0]+1)), classes)
        plt.ylabel(ft)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if saveto:
            plt.savefig(saveto/ft)
        if close_after_plotting:
            plt.close()
    return


#%%% mRMR feature selection
def mRMR_select(feat_set, redundancy, relevance, criterion='MIQ'):

    # index in the enntire feature vector
    index = np.arange(relevance.shape[0])
    index = np.delete(index, feat_set)

    # redunduncy component
    c_i = np.mean(redundancy[:, np.array(feat_set).reshape(-1)], axis=1)
    c_i = np.delete(c_i, feat_set)

    # relevance component
    t_i = np.delete(relevance, feat_set)

    if criterion=='MIQ':
        max_score = np.max(t_i/(c_i+0.01))
        select = index[np.argmax(t_i/(c_i+0.01))]
    elif criterion=='MID':
        max_score = np.max(t_i-c_i)
        select = index[np.argmax(t_i-c_i)]

    return np.append(feat_set, select), max_score

def _calc_MI(x, y, n_bins):
    from sklearn.metrics import mutual_info_score
    from sklearn.preprocessing import KBinsDiscretizer

    encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

    c_xy = np.zeros((x.shape[0], 2)).astype(int)

    mask = ~np.isnan(x)
    c_xy[mask, 0] = encoder.fit_transform(x[mask].reshape(-1,1)).reshape(-1)
    c_xy[~mask, 0] = n_bins
    mask = ~np.isnan(y)
    c_xy[mask, 1] = encoder.fit_transform(y[mask].reshape(-1,1)).reshape(-1)
    c_xy[~mask, 1] = n_bins

    return mutual_info_score(c_xy[:,0], c_xy[:,1])

def get_redundancy(feat, redundancy_func='mutual_info', get_abs=False, n_bins=5):
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.metrics import mutual_info_score

    if isinstance(redundancy_func, str):
        if redundancy_func.endswith('_corr'):
            method = redundancy_func.split('_')[0]
            return pd.DataFrame(feat).corr(method=method).abs().values

        elif redundancy_func == 'mutual_info':
            encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            feat_discrete = encoder.fit_transform(feat)

            n_ft = feat.shape[1]
            redundancy = np.zeros((n_ft, n_ft))
            for i in range(n_ft):
                for j in range(i, n_ft):
                    redundancy[i,j] = mutual_info_score(
                        np.array(feat_discrete)[:,i], np.array(feat_discrete)[:,j])
                    redundancy[j,i] = redundancy[i,j]
            return redundancy

        else:
            raise ValueError(
                """Name passed to redundancy_func not recognized. See function
                docstring for available options.
                """)

    else:
        if not callable(redundancy_func):
            raise ValueError('Data type of redundancy_func not recognized.')

    n_ft = feat.shape[1]
    redundancy = np.zeros((n_ft, n_ft))
    for i in range(n_ft):
        for j in range(i, n_ft):
            redundancy[i,j] = redundancy_func(np.array(feat)[:,i], np.array(feat)[:,j])
            redundancy[j,i] = redundancy[i,j]

    if get_abs:
        redundancy = np.abs(redundancy)

    return redundancy


def get_relevance(feat, y_class, relevance_func='mutual_info'):
    from sklearn.feature_selection import \
        chi2, f_classif, mutual_info_classif
    from scipy.stats import kruskal

    feat = np.array(feat)

    if isinstance(relevance_func, str):
        if relevance_func == 'f_classif':
            relevance, _ = f_classif(feat, y_class)
        elif relevance_func == 'chi2':
            relevance, _ = chi2(feat, y_class)
        elif relevance_func == 'mutual_info':
            relevance = mutual_info_classif(feat, y_class)
        elif relevance_func == 'kruskal':
            relevance = np.zeros(feat.shape[1])
            for i, ft in enumerate(feat.T):
                relevance[i], _ = kruskal(
                    *[ft[y_class==iy] for iy in np.unique(y_class)])
    else:
        feat = np.array(feat)
        relevance = np.zeros(feat.shape[1])
        for i in range(feat.shape[1]):
            relevance[i] = relevance_func(feat[:,i], y_class)

    return relevance

