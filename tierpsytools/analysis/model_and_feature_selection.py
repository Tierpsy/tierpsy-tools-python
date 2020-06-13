#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:33:54 2020

@author: em812
"""

def RFE_selection(X, y, n_feat, estimator, step=100, save_to=None):
    from sklearn.feature_selection import RFE
    from time import time
    import pickle

    print('RFE selection for n_feat={}.'.format(n_feat))
    start_time = time()

    rfe = RFE(estimator, n_features_to_select=n_feat, step=step)
    X_sel = rfe.fit_transform(X,y)

    print("RFE: --- %s seconds ---" % (time() - start_time))

    if save_to is not None:
        pickle.dump( rfe, open(save_to/'fitted_rfe_nfeat={}.p'.format(n_feat), "wb") )

    return X_sel, rfe.support_, rfe

def kbest_selection(X, y, n_feat, score_func=None):
    from sklearn.feature_selection import SelectKBest, f_classif

    if score_func is None:
        score_func = f_classif

    selector = SelectKBest(score_func=score_func, k=n_feat)
    X_sel = selector.fit_transform(X, y)

    return X_sel, selector.support_


def model_selection(X, y, estimator, param_grid, cv_strategy=0.2, save_to=None, saveid=None):
    from sklearn.model_selection import GridSearchCV
    from time import time
    import pickle

    print('Starting grid search CV...')
    start_time = time()
    grid_search = GridSearchCV(
        estimator, param_grid=param_grid, cv=cv_strategy, n_jobs=-1, return_train_score=True)

    grid_search.fit(X, y)
    print("Grid search: --- %s seconds ---" % (time() - start_time))

    if save_to is not None:
        pickle.dump( grid_search, open(save_to/'fitted_gridsearchcv_nfeat={}.p'.format(saveid), "wb") )

    return grid_search.best_estimator_, grid_search.best_score_


def get_feat_sets_RFECV(rfecv, ):
    import numpy as np
    n_features = rfecv.support_.shape[0]
    step=rfecv.step

    n_feat = [n_features]
    n_features_to_select = rfecv.min_features_to_select

    while n_feat[-1] > n_features_to_select:
        threshold = min(step, n_feat[-1] - n_features_to_select)
        n_feat.append(n_feat[-1]-threshold)

    return np.flip(n_feat)




