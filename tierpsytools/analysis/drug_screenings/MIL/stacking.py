#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:33:38 2020

@author: em812
"""
import numpy as np
import pandas as pd
from tierpsytools.feature_processing.scaling_class import scalingClass
from sklearn.metrics import accuracy_score
from tierpsytools.analysis.drug_screenings.MIL.majority_vote import score_majority_vote, get_majority_vote
import pdb

def get_sum_of_votes(group, labels, y_pred=None, probas=None, sum_type='counts'):
    from collections import Counter
    from sklearn.preprocessing import LabelEncoder

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

    return pred


def _cv_predict_parallel(
        X, y,
        splitter, estimator,
        group=None, scale_function=None,
        n_jobs=-1
        ):
    from joblib import Parallel, delayed

    def _one_fit(X, y, group, train_index, test_index, estimator, scale_function=None):
        # Normalize
        scaler = scalingClass(scaling=scale_function)
        X_train = scaler.fit_transform(X[train_index])
        X_test = scaler.transform(X[test_index])

        # Train classifier
        estimator.fit(X_train, y[train_index])

        # Predict
        y_pred = estimator.predict(X_test)
        labels = estimator.classes_
        if hasattr(estimator, 'predict_proba'):
            y_probas = estimator.predict_proba(X_test)
        elif hasattr(estimator, 'decision_function'):
            y_probas = estimator.decision_function(X_test)
        elif hasattr(estimator, 'oob_decision_function') and estimator.oob_score:
            y_probas = estimator.oob_decision_function(X_test)
        else:
            y_probas = None
        return test_index, y_pred, y_probas, labels, estimator

    pred = np.empty_like(y)
    probas = np.empty((X.shape[0], np.unique(y).shape[0]))

    parallel = Parallel(n_jobs=-1, verbose=True)
    func = delayed(_one_fit)

    res = parallel(
        func(X, y, group, train_index, test_index, estimator, scale_function=scale_function)
        for train_index, test_index in splitter.split(X, y, group))

    for  test_index,y_pred,y_probas,_,_ in res:
        pred[test_index] = y_pred
        if y_probas is not None:
            probas[test_index] = y_probas
    labels = [lab for _,_,_,lab,_ in res]
    assert all([ all(labels[0]==labels[i]) for i in range(1,len(labels)) ])
    trained_estimators = [est for _,_,_,_,est in res]

    return pred, probas, labels[0], trained_estimators

def _cv_predict(
        X, y,
        splitter, estimator,
        group=None, scale_function=None,
        n_jobs=-1
        ):

    pred = np.empty_like(y)
    probas = np.empty((X.shape[0], np.unique(y).shape[0]))

    trained_estimators = []
    labels = []
    for train_index, test_index in splitter.split(X, y, group):

        # Normalize
        scaler = scalingClass(scaling=scale_function)
        X_train = scaler.fit_transform(X[train_index])
        X_test = scaler.transform(X[test_index])

        # Train classifier
        estimator.fit(X_train, y[train_index])
        trained_estimators.append(estimator)

        # Predict
        pred[test_index] = estimator.predict(X_test)
        labels.append(estimator.classes_)
        if hasattr(estimator, 'predict_proba'):
            probas[test_index] = estimator.predict_proba(X_test)
        elif hasattr(estimator, 'decision_function'):
            probas[test_index] = estimator.decision_function(X_test)
        elif hasattr(estimator, 'oob_decision_function') and estimator.oob_score:
            probas[test_index] = estimator.oob_decision_function(X_test)
        else:
            probas = None

        assert all([ all(labels[0]==labels[i]) for i in range(1,len(labels)) ])

    return pred, probas, labels[0], trained_estimators

def _get_cv_predictions(
        X, y,
        splitter, estimator,
        group=None, scale_function=None,
        n_jobs=-1
        ):

    if n_jobs==1:
        return _cv_predict(X, y, splitter, estimator, group=group, scale_function=scale_function)
    else:
        return _cv_predict_parallel(X, y, splitter, estimator, group=group, scale_function=scale_function, n_jobs=n_jobs)


def _get_majority_vote(group, labels, y_pred=None, probas=None, sum_rule='counts'):
    """
    Wrapper function to get the majority vote in the form required for stacking
    """
    y_maj = get_majority_vote(group, y_pred=y_pred, probas=probas, labels=labels, sum_rule=sum_rule)
    return pd.DataFrame(y_maj, index=['pred']).T


def strain_stack_majority_vote_CV(
        Xs, ys, groups,
        base_estimator, stacked_estimator,
        splitter, stacked_splitter, pred_type='counts',
        scale_function=None,
        stacked_scale_function=None,
        retrain_estimators=False,
        n_jobs=-1):

    ## CV with base models initial datasets
    #--------------------------
    pred_votes = {}
    scores = {}
    scores_group = {}
    trained_estimators = {}
    for strain in Xs.keys():
        X = np.array(Xs[strain])
        y = np.array(ys[strain])
        group = np.array(groups[strain])

        pred, probas, labels, trained_estimators[strain] = \
            _get_cv_predictions(
                X, y, splitter, base_estimator, group=group,
                scale_function=scale_function, n_jobs=n_jobs)

        # Majority vote
        if pred_type=='single_pred':
            pred_votes[strain] = _get_majority_vote(group, labels, y_pred=pred, probas=probas)
        else:
            pred_votes[strain] = get_sum_of_votes(group, labels, y_pred=pred, probas=probas, sum_type=pred_type)

        scores_group[strain] = score_majority_vote(y, group, y_pred=pred)
        scores[strain] = accuracy_score(y, pred)


    ## Stack
    #--------
    for strain in pred_votes.keys():
        pred_votes[strain] = pred_votes[strain].rename(
            columns={col:'_'.join([str(col),strain]) for col in pred_votes[strain].columns})

    pred_votes = pd.concat([item for item in pred_votes.values()], axis=1)

    # Get the target labels for each group
    y = np.concatenate([val for val in ys.values()])
    group = np.concatenate([val for val in groups.values()])

    y_group = pd.DataFrame(y).groupby(by=group).apply(lambda x: np.unique(x))
    assert all([len(x)==1 for x in y_group.values])
    y_group = y_group.apply(lambda x:x[0])

    y_group == y_group.loc[pred_votes.index].values


    ## CV with majority vote stacked predictions
    #--------------------------
    pred, probas, labels, trained_estimators['stacked'] = \
        _get_cv_predictions(
            pred_votes.values, y_group, stacked_splitter, stacked_estimator,
            scale_function=stacked_scale_function, n_jobs=n_jobs)
    scores_group['stacked'] = accuracy_score(y_group, pred)

    ## Train the base estimatores and the stacked_estimator with the entire dataset
    #--------------------------
    if retrain_estimators:
        # base
        for strain,X in Xs.items():
            base_estimator.fit(X,y)
            trained_estimators[strain] = base_estimator
        # stacked
        stacked_estimator.fit(pred_votes, y_group)
        trained_estimators['stacked'] = stacked_estimator

    return scores, scores_group, trained_estimators

