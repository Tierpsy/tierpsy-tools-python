#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:33:38 2020

@author: em812
"""
import numpy as np
import pandas as pd
from tierpsytools.analysis.drug_screenings.MIL.majority_vote import \
    majority_vote_CV, get_majority_vote, get_sum_of_votes, _get_y_group
from tierpsytools.analysis.classification_tools import cv_score
import pdb


def strain_stack_majority_vote_CV(
        Xs, ys, groups,
        base_estimator, stacked_estimator,
        splitter, stacked_splitter,
        vote_type='counts', pred_type='counts',
        scale_function=None,
        stacked_scale_function=None,
        retrain_estimators=False,
        n_jobs=-1, scorer=None):

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

        _scores, _scores_maj, pred, probas, labels = majority_vote_CV(
            X, y, group, base_estimator, splitter,
            scale_function=scale_function, n_jobs=n_jobs, vote_type=vote_type,
            return_predictions=True, scorer=scorer
            )

        scores[strain] = _scores
        scores_group[strain] = _scores_maj

        # Majority vote
        if pred_type=='single_pred':
            pred_votes[strain] = get_majority_vote(group, labels, y_pred=pred, probas=probas)
        else:
            pred_votes[strain] = get_sum_of_votes(group, labels, y_pred=pred, probas=probas, sum_type=pred_type)



    ## Stack
    #--------
    for strain in pred_votes.keys():
        pred_votes[strain] = pred_votes[strain].rename(
            columns={col:'_'.join([str(col),strain]) for col in pred_votes[strain].columns})

    pred_votes = pd.concat([item for item in pred_votes.values()], axis=1)

    # Get the target labels for each group
    y = np.concatenate([val for val in ys.values()])
    group = np.concatenate([val for val in groups.values()])

    y_group = _get_y_group(y, group)

    y_group == y_group.loc[pred_votes.index].values


    ## CV with majority vote stacked predictions
    #--------------------------
    scores_group['stacked'] = cv_score(
        pred_votes.values, y_group,
        stacked_splitter, stacked_estimator,
        group=None, sample_weight=None,
        scale_function=stacked_scale_function,
        n_jobs=n_jobs, scorer=scorer)

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

