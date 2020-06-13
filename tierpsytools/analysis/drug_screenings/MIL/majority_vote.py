#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:01:33 2020

@author: em812
"""
import numpy as np

def get_two_most_likely_accuracy(ytest, ytest_pred_two):

    check=[]
    for i,y in enumerate(ytest):
        if y in ytest_pred_two[i]:
            check.append(True)
    acc = np.sum(check)/len(check)
    return acc

def get_two_most_likely(Xtest, estimator):

    ytest_pred_two = np.empty([Xtest.shape[0],2])

    ytest_pred_proba = estimator.predict_proba(Xtest)
    indx = np.flip(np.argsort(ytest_pred_proba,axis=1),axis=1)
    classes = estimator.classes_
    for cmpd in range(Xtest.shape[0]):
        ytest_pred_two[cmpd] = classes[indx[cmpd]][0:2]

    return ytest_pred_two

def get_seen_compounds(Xtest, ytest, ytrain):
    """
    Keep only test set cpmpounds that belong to MOAs that were seen in ytrain
    """
    import pandas as pd
    if isinstance(ytest,list):
        ytest=np.array(ytest)

    if isinstance(Xtest,pd.DataFrame):
        Xtest=Xtest.values
    seen = [i for i,y in enumerate(ytest) if y in ytrain]
    ytest = ytest[seen]
    Xtest = Xtest[seen,:]

    return Xtest,ytest,seen

def majority_vote_LeaveOneBagOut_CV(
        X, y, group, estimator, scale_function=None):
    from tierpsytools.feature_processing.scaling_class import scalingClass
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import LeaveOneOut
    ## Majority vote
    #---------------
    splitter = LeaveOneOut()

    # Make feature matrix with bags of sample points per compound
    ugroups = np.unique(group)

    Xb = np.array([X[group==grp] for grp in ugroups])
    yb = np.array([y[group==grp] for grp in ugroups])
    groupb = np.array([group[group==grp] for grp in ugroups])

    scores = []
    scores_maj = []
    for train_index, test_index in splitter.split(Xb, yb):
        X_train = np.concatenate(Xb[train_index])
        X_test = np.concatenate(Xb[test_index])
        y_train = np.concatenate(yb[train_index])
        y_test = np.concatenate(yb[test_index])
        group_test = np.concatenate(groupb[test_index])

        # Normalize
        scaler = scalingClass(scaling=scale_function)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train classifier
        estimator.fit(X_train,y_train)

        # Predict
        y_pred = estimator.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
        scores_maj.append(score_majority_vote(y_test, y_pred, group_test))

    return scores, scores_maj

def majority_vote_CV(
        X, y, group, estimator, splitter, scale_function=None):
    from tierpsytools.feature_processing.scaling_class import scalingClass
    from sklearn.metrics import accuracy_score
    ## Majority vote
    #---------------
    X = np.array(X)
    y = np.array(y)
    group = np.array(group)

    scores = []
    scores_maj = []
    for train_index, test_index in splitter.split(X, y, group):

        # Normalize
        scaler = scalingClass(scaling=scale_function)
        X_train = scaler.fit_transform(X[train_index])
        X_test = scaler.transform(X[test_index])

        # Train classifier
        estimator.fit(X_train, y[train_index])

        # Predict
        y_pred = estimator.predict(X_test)
        scores.append(accuracy_score(y[test_index], y_pred))
        scores_maj.append(
            score_majority_vote(y[test_index], y_pred, group[test_index]))

    return scores, scores_maj


def _one_fit(X, y, group, train_index, test_index, estimator, scale_function=None):
    from tierpsytools.feature_processing.scaling_class import scalingClass
    from sklearn.metrics import accuracy_score

    # Normalize
    scaler = scalingClass(scaling=scale_function)
    X_train = scaler.fit_transform(X[train_index])
    X_test = scaler.transform(X[test_index])

    # Train classifier
    estimator.fit(X_train, y[train_index])

    # Predict
    y_pred = estimator.predict(X_test)
    score = accuracy_score(y[test_index], y_pred)
    score_maj = score_majority_vote(y[test_index], y_pred, group[test_index])
    return score, score_maj

def majority_vote_CV_parallel(
        X, y, group, estimator, splitter, scale_function=None, n_jobs=-1):
    from joblib import Parallel, delayed

    ## Majority vote
    #---------------
    X = np.array(X)
    y = np.array(y)
    group = np.array(group)

    parallel = Parallel(n_jobs=-1, verbose=True)
    func = delayed(_one_fit)

    scores = parallel(
        func(X, y, group, train_index, test_index, estimator, scale_function=scale_function)
        for train_index, test_index in splitter.split(X, y, group))

    scores_maj = [maj for st, maj in scores]
    scores = [st for st, maj in scores]

    return scores, scores_maj

def get_majority_vote(y_pred, group, probas=None, labels=None):
    """
    Estimate the classification accuracy when groups of data points are classified together based on a majority vote.
    param:
        y_true: the true class labels of the samples (array size n_samples)
        y_pred: the predicted class labels from the classfier (array size n_samples)
        groups: an array defining the groups of data points (array size n_samples)
    """
    from collections import Counter

    y_maj = {}

    for grp in np.unique(group):

        c = Counter(y_pred[group==grp])

        #value,count = c.most_common()[0]
        counts=np.array([votes for clss,votes in c.most_common()])

        if (sum(counts==counts[0])==1) or (probas is None):
            value,count = c.most_common()[0]
            y_maj[grp] = value

        else:   # if more than one labels have the same number of votes and we have proba info
            assert len(labels)==probas.shape[1]
            values = np.array([clss for clss,votes in c.most_common()])
            equal_classes = values[counts==counts[0]]
            probas_of_equal_classes = []
            for iclass in equal_classes:
                probas_of_equal_classes.append(
                    np.mean(probas[group==grp, labels==iclass]))
            most_likely_class = equal_classes[np.argmax(probas_of_equal_classes)]
            y_maj[grp] = most_likely_class
    return y_maj

def score_majority_vote(y_real, y_pred, group, probas=None, labels=None):
    from sklearn.metrics import accuracy_score

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)

    ugroups = np.unique(group)

    assert np.all([np.unique(y_real[group==g]).shape[0]==1 for g in ugroups]), \
        'The real labels are not unique per group.'

    y_group = [np.unique(y_real[group==g])[0] for g in ugroups]
    y_maj = get_majority_vote(y_pred, group, probas=probas, labels=labels)
    y_maj = [y_maj[key] for key in ugroups]

    return accuracy_score(y_group, y_maj)


def get_two_most_likely_majority_vote(y_pred,groups):
    """
    Estimate the classification accuracy when groups of data points are classified together based on a majority vote.
    param:
        y_true: the true class labels of the samples (array size n_samples)
        y_pred: the predicted class labels from the classfier (array size n_samples)
        groups: an array defining the groups of data points (array size n_samples)
    """
    from collections import Counter

    y_maj = np.empty((y_pred.shape[0],2))

    for grp in np.unique(groups):

        c = Counter(y_pred[groups==grp])

        if len(c.most_common())>1:
            for rnk in [0,1]:
                value,count = c.most_common()[rnk]
                y_maj[groups==grp,rnk] = value
        else:
            value,count = c.most_common()[0]
            y_maj[groups==grp,:] = [value,value]

        # if more than one labels have the same number of votes
        if len(c.most_common())>1 and c.most_common()[0][1]==c.most_common()[1][1]:
            print('Warning: the samples of compound {} are classified in more than one classes with the same frequency.'.format(grp))

    return y_maj
