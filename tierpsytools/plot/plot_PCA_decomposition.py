#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:25:49 2020

@author: em812
"""


def plot_pca(
        x, n_dim=2, which_PCs=None, labels=None, savefig=None, closefig=False,
        title=None, add_legend=False, colors=None, **scatter_kwargs):
    """
    Plot data in PCA space in 2 or 3 dimensions. Can chose to plot any
    combination of components with which_PCs or plot the first two or three
    components (by default)

    Parameters
    ----------
    x : array shape = (n_datapoints, n_features)
        data.
    n_dim : int, optional
        Number of dimension of the plot. The default is 2.
    color : array shape = (n_datapoints,), optional
        Colors to apply to the datapoints. The default is None.
    savefig : path, optional
        Path to save the figure. The default is None.
    closefig : bool, optional
        If true, the figure will close before the function returns.
        The default is False.

    Raises
    ------
    ValueError
        When the n_dim is anything other than 2 or 3.

    Returns
    -------
    None.

    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    from math import ceil

    if n_dim<2 or n_dim>3:
        raise ValueError('Only 2 or 3 dimensions can be plotted.')

    if colors is None and labels is not None:
        n_labels = np.unique(labels).shape[0]
        colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
            '#a9a9a9', '#9A6324','#469990','#ffd8b1','#f032e6', '#fabebe',
            '#e6beff', '#800000', '#aaffc3', '#808000', '#000075', '#808080',
            '#000000'
            ]
        if n_labels>len(colors):
            colors = colors*ceil(n_labels/len(colors))

    if which_PCs is None:
        n_components = n_dim
        pcs = list(range(n_dim))
    else:
        n_components = max(which_PCs)+1
        pcs = which_PCs

    pca = PCA(n_components=n_components)
    Y = pca.fit_transform(x)
    Y = Y[:,pcs]

    if n_dim==2:
        if labels is None:
            plt.scatter(*Y.T, **scatter_kwargs)
        else:
            for igrp,group in enumerate(np.unique(labels)):
                plt.scatter(*Y[labels==group,:].T,label=group,c=colors[igrp],
                            **scatter_kwargs)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        if title is not None:
            plt.title(title)
        if add_legend:
            plt.legend()
        if savefig is not None:
            plt.savefig(savefig)
        if closefig:
            plt.close()
    elif n_dim==3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if labels is None:
            ax.scatter(*Y.T, **scatter_kwargs)
        else:
            for igrp,group in enumerate(np.unique(labels)):
                ax.scatter(*Y[labels==group,:].T,label=group,c=colors[igrp],
                           **scatter_kwargs)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        if add_legend:
            plt.legend()
        if title is not None:
            plt.title(title)
        if savefig is not None:
            plt.savefig(savefig)
        if closefig:
            plt.close()

    return
