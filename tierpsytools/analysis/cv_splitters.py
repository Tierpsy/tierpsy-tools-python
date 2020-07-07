#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:05:37 2020

author: em812
based on:
https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
"""

import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection._split import _BaseKFold
from sklearn.preprocessing import LabelEncoder
import pdb

class StratifiedGroupKFold(_BaseKFold):
    """
    Makes stratified folds according to the class labels y, but each group
    defined in groups, is assigned in one fold (all the points of the group
                                                are assigned to the same fold)
    """
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y, groups, seed=None):
        group_counts = pd.DataFrame(
            {'y':y, 'groups':groups}).groupby(by='y').agg({"groups": "nunique"})
        if np.any(group_counts.values<self.n_splits):
            raise ValueError('Some of the classes have less groups than '+
                             'the number of splits.')

        y = LabelEncoder().fit_transform(y)
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices


if __name__=="__main__":

    groups = [1,1,2,2,3,3,4,4]
    y=[1,1,1,1,0,0,0,0]
    X=np.random.rand(8,2)
    k=2

    splitter=StratifiedGroupKFold(n_splits=k)

    for train_ind, test_ind in splitter.split(X,y,groups):
        print(train_ind, test_ind)