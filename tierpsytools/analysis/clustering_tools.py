#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:49:55 2020

@author: em812
"""


import numpy as np
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, fcluster


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def hierarchical_purity(data, labels,
                        linkage_matrix=None, linkage_method='average',
                        criterion='distance', n_random=1000
                        ):

    if linkage_matrix is None:
        linkage_matrix = linkage(data, method=linkage_method)

    distances = np.unique(linkage_matrix[:, -2])

    purity = []
    purity_rand = []
    n_clusters = []
    for dist in distances:
        cl_ids = fcluster(linkage_matrix, t=dist, criterion='distance')
        purity.append(purity_score(labels, cl_ids))

        # Random shuffling
        n_clust = np.unique(cl_ids).shape[0]

        p_rand = []
        for i in range(n_random):
            rand_cl_ids = np.random.randint(0, n_clust, size=data.shape[0])
            p_rand.append(purity_score(labels, rand_cl_ids))

        purity_rand.append(p_rand)
        n_clusters.append(n_clust)

    return distances, n_clusters, np.array(purity), np.array(purity_rand)


if __name__=="__main__":

    y_true = [1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3]
    y_pred = [0,0,0,0,0,1,2,2,0,1,1,1,1,1,2,2,2]

    pur = purity_score(y_true, y_pred)
    print(pur)
