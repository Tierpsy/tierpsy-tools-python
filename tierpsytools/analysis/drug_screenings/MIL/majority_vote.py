#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:01:33 2020

@author: em812
"""
import numpy as np
import pandas as pd
from tierpsytools.analysis.classification_tools import cv_predict
from tierpsytools.feature_processing.scaling_class import scalingClass
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tierpsytools.analysis.helper import _get_multi_sclassifscorers
import pdb

def _get_y_group(y, group):

    y_group = pd.DataFrame(y).groupby(by=group).apply(lambda x: np.unique(x))

    assert all([len(x)==1 for x in y_group.values]), 'y is not unique in each group'

    y_group = y_group.apply(lambda x:x[0])

    return y_group

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
    if isinstance(ytest,list):
        ytest=np.array(ytest)

    if isinstance(Xtest,pd.DataFrame):
        Xtest=Xtest.values
    seen = [i for i,y in enumerate(ytest) if y in ytrain]
    ytest = ytest[seen]
    Xtest = Xtest[seen,:]

    return Xtest,ytest,seen

# %% Legacy
def _majority_vote_CV(
        X, y, group, estimator, splitter, scale_function=None):

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
            score_majority_vote(y[test_index], group[test_index]), y_pred=y_pred)

    return scores, scores_maj

def majority_vote_CV_parallel(
        X, y, group, estimator, splitter,
        scale_function=None, n_jobs=-1, sum_rule='counts'
        ):

    ## Majority vote
    #---------------
    def _one_fit(X, y, group, train_index, test_index, estimator, scale_function=None):
        # Normalize
        scaler = scalingClass(scaling=scale_function)
        X_train = scaler.fit_transform(X[train_index])
        X_test = scaler.transform(X[test_index])

        # Train classifier
        estimator.fit(X_train, y[train_index])

        # Predict
        y_pred = estimator.predict(X_test)
        if hasattr(estimator, 'predict_proba'):
            probas = estimator.predict_proba(X_test)
        elif hasattr(estimator, 'decision_function'):
            probas = estimator.decision_function(X_test)
        elif hasattr(estimator, 'oob_decision_function') and estimator.oob_score:
            probas = estimator.oob_decision_function(X_test)
        else:
            probas = None
        score = accuracy_score(y[test_index], y_pred)
        score_maj = score_majority_vote(
            y[test_index], group[test_index], y_pred=y_pred,
            probas=probas, labels=estimator.classes_, vote_type=sum_rule)
        return score, score_maj

    X = np.array(X)
    y = np.array(y)
    group = np.array(group)

    parallel = Parallel(n_jobs=-1, verbose=True)
    func = delayed(_one_fit)

    scores = parallel(
        func(X, y, group, train_index, test_index, estimator, scale_function=scale_function)
        for train_index, test_index in splitter.split(X, y, group))

    scores_maj = [maj for _, maj in scores]
    scores = [st for st, _ in scores]

    return scores, scores_maj

#%%

def majority_vote_CV(
        X, y, group, estimator, splitter,
        vote_type='counts', sample_weight = None,
        scale_function=None, n_jobs=-1,
        return_predictions=False, scorer=None
        ):

    scorers = _get_multi_sclassifscorers(scorer)

    pred, probas, labels, test_folds, trained_estimators = cv_predict(
        X, y, splitter, estimator, group=group, scale_function=scale_function,
        n_jobs=n_jobs)

    scores = {key:[] for key in scorers.keys()}
    scores_maj = {key:[] for key in scorers.keys()}
    for test_index in test_folds:
        if probas is not None:
            _probas = probas[test_index]
        else:
            _probas = None
        if sample_weight is not None:
            _weights = sample_weight[test_index]
        else:
            _weights = None

        for key in scorers.keys():
            scores[key].append(scorers[key].score(
                y[test_index], pred=pred[test_index], probas=_probas,
                  labels=labels, sample_weight=_weights
                ))
            scores_maj[key].append(scorers[key].score_maj(
                y[test_index], group[test_index],
                pred=pred[test_index], probas=_probas, labels=labels,
                sample_weight=_weights, vote_type=vote_type
                ))


    if return_predictions:
        return scores, scores_maj, pred, probas, labels
    else:
        return scores, scores_maj


def get_majority_vote(
        group, y_pred=None, probas=None, labels=None, vote_type='counts'):
    """
    Get the majority vote predictions per group.
    param:
        groups: an array defining the groups of data points (array size n_samples)
        y_pred: the predicted class labels for each data point (array size n_samples)
        probas: the probabilities for each class for each data point (array shape=(n_samples, n_classes) )
        labels: the class labels that match each column of the probas array
        vote_type: the rule the the majoroty vote is based on ['counts', 'probas']
    """

    if probas is not None and labels is None:
        raise ValueError('Must provide class labels corresponding to the '+
                         'columns of the probas.')

    if vote_type == 'counts' and y_pred is None:
        if probas is None:
            raise ValueError('Must provide either y_pred or probas to use the counts sum rule.')
        else:
            y_pred = [labels[maxind] for maxind in np.argmax(probas, axis=1)]
            y_pred = np.array(y_pred)

    if vote_type == 'counts' and labels is not None:
        assert all([pred in labels for pred in y_pred]), \
            'The predictions in y_pred do not match the labels provided.'

    if vote_type == 'counts':
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
        y_maj = pd.Series(y_maj)
    elif vote_type == 'probas':
        labels = np.array(labels).reshape(-1)
        assert labels.shape[0]==probas.shape[1]
        group_probas = pd.DataFrame(probas).groupby(by=group).sum()
        y_maj = pd.Series({grp:labels[np.argmax(group_probas.loc[grp,:].values)]
                           for grp in np.unique(group)})

    return y_maj


def get_sum_of_votes(group, labels, y_pred=None, probas=None,
                     sum_type='counts', normalized=False):

    # Checks
    if y_pred is not None:
        assert all([pred in labels for pred in y_pred]), \
            'The predictions in y_pred do not match the labels provided.'

    if sum_type == 'counts' and y_pred is None:
        if probas is None:
            raise ValueError('Must provide either y_pred or probas to use the counts sum_type.')
        else:
            y_pred = [labels[maxind] for maxind in np.argmax(probas, axis=1)]
            y_pred = np.array(y_pred)

    if sum_type=='probas' and probas is None:
        raise ValueError('Must provide probas to use the probas sum_type.')

    # Get the sum of votes
    encoder_lab = LabelEncoder()
    encoder_group = LabelEncoder()
    if y_pred is not None:
        labels = encoder_lab.fit_transform(labels)
        y_pred = encoder_lab.transform(y_pred)
    group = encoder_group.fit_transform(group)
    ugroup = np.unique(group)


    if sum_type == 'counts':
        pred = np.zeros((ugroup.shape[0], labels.shape[0]))
        for grp in ugroup:
            c = Counter(y_pred[group==grp])
            for clss,value in c.items():
                pred[grp, clss] = value
        pred = pd.DataFrame(pred, index=encoder_group.classes_, columns=encoder_lab.classes_)

    elif sum_type == 'probas':
        pred = pd.DataFrame(
            probas).groupby(by=group).sum()
        pred.index = encoder_group.classes_
        pred.columns = encoder_lab.classes_

    if normalized:
        pred = pd.DataFrame(normalize(pred, norm='l1', axis=1),
                            columns=pred.columns, index=pred.index)

    return pred


def score_majority_vote(
        y_real, group,
        y_pred=None, probas=None, labels=None,
        vote_type='counts',
        scorer=None
        ):

    if scorer is None:
        score_func = accuracy_score
    else:
        score_func = scorer

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)

    ugroups = np.unique(group)

    assert np.all([np.unique(y_real[group==g]).shape[0]==1 for g in ugroups]), \
        'The real class labels are not unique per group.'

    y_group = [np.unique(y_real[group==g])[0] for g in ugroups]
    y_maj = get_majority_vote(
        group, y_pred=y_pred, probas=probas, labels=labels, vote_type=vote_type)
    y_maj = y_maj[ugroups]

    return score_func(y_group, y_maj)


def get_two_most_likely_majority_vote(y_pred,groups):
    """
    Estimate the classification accuracy when groups of data points are classified together based on a majority vote.
    param:
        y_true: the true class labels of the samples (array size n_samples)
        y_pred: the predicted class labels from the classfier (array size n_samples)
        groups: an array defining the groups of data points (array size n_samples)
    """

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
