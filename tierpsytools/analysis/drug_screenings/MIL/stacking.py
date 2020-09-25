#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:33:38 2020

@author: em812
"""
import numpy as np
import pandas as pd
from tierpsytools.analysis.classification_tools import cv_predict
from tierpsytools.analysis.drug_screenings.MIL.majority_vote import \
    majority_vote_CV, get_majority_vote, get_sum_of_votes, _get_y_group
from tierpsytools.analysis.classification_tools import cv_score
from tierpsytools.analysis.helper import _get_multi_sclassifscorers
import pdb
from sklearn.base import clone

class StackNode():
    def __init__(self, estimator, splitter, is_grouped_data=True,
                 vote_type=None, scale_function=None,
                 n_jobs=-1, scorer=None):

        self.estimator = clone(estimator)
        self.splitter = splitter
        self.is_grouped_data = is_grouped_data
        if is_grouped_data:
            if vote_type is None:
                raise ValueError('Define vote_type to get the group-level accuracy.')
            else:
                self.vote_type = vote_type
        else:
            self.vote_type = None

        self.scale = scale_function
        self.n_jobs = n_jobs
        self.scorers = _get_multi_sclassifscorers(scorer)

    def fit(self, X, y, group=None, sample_weight=None):
        X = np.array(X)
        y = np.array(y)
        self.y = y

        if sample_weight is None:
            sample_weight = np.ones(y.shape)
        else:
            sample_weight = np.array(sample_weight)

        # Check data
        if self.is_grouped_data and group is None:
            raise ValueError('This class instance is defined for grouped data.'+
                             'Must provide group labels to fit.')
        if not self.is_grouped_data and group is not None:
            raise ValueError('This class instance is defined for independent data.'+
                             'The group labels cannot be used.')

        # Store group-related info
        if group is not None:
            group = np.array(group)
            self.group = group
            self.y_group = _get_y_group(y, group)

        # CV predictions
        self.y_pred, self.probas, self.labels, test_folds, _ = cv_predict(
            X, y, self.splitter, self.estimator, group=group,
            scale_function=self.scale, n_jobs=self.n_jobs,
            sample_weight=sample_weight)

        # Get group-level predictions
        if self.is_grouped_data:
            self.group_pred = get_majority_vote(
                group, y_pred=self.y_pred, probas=self.probas,
                labels=self.labels, vote_type=self.vote_type)

        self.estimator = self.estimator.fit(X,y)

        # Score per fold
        self.scores = {key:[] for key in self.scorers.keys()}
        if self.is_grouped_data:
            self.scores_group = {key:[] for key in self.scorers.keys()}
        for test_index in test_folds:
            if self.probas is not None:
                _probas = self.probas[test_index]
            else:
                _probas = None
            for key in self.scorers.keys():
                self.scores[key].append(self.scorers[key].score(
                    y[test_index], pred=self.y_pred[test_index],
                    probas=_probas, labels=self.labels,
                    sample_weight=sample_weight[test_index]
                    ))
                if self.is_grouped_data:
                    self.scores_group[key].append(self.scorers[key].score_maj(
                        y[test_index], group[test_index],
                        pred=self.y_pred[test_index], probas=_probas,
                        labels=self.labels, vote_type=self.vote_type,
                        sample_weight=sample_weight[test_index]
                        ))



    def fit_transform(self, X, y, group=None, sample_weight=None,
                      pred_type='single_pred', group_pred=False):

        if group_pred and not self.is_grouped_data:
            raise ValueError('Cannot give grouped predictions for data that'+
                             'are not grouped.')

        self.fit(X, y, group=group, sample_weight=sample_weight)

        return self.get_targets_predictions(pred_type, group_pred)

    def get_targets_predictions(self, pred_type, group_pred):
        if not group_pred and pred_type=='counts':
            raise ValueError('The option counts for pred_type is not supported'+
                             'for predictions that are not grouped.')

        if pred_type == 'probas':
            if self.probas is None:
                raise Exception('There are no class probabilities estimates.')

        if group_pred:
            targets = self.y_group
            if pred_type=='single_pred':
                predictions = self.group_pred
            else:
                predictions = get_sum_of_votes(
                    self.group, self.labels, y_pred=self.y_pred, probas=self.probas, sum_type=pred_type)
        else:
            targets = self.y
            if pred_type=='single_pred':
                predictions = self.y_pred
            elif pred_type=='probas':
                predictions = self.probas

        return targets, predictions


def l2_stack_then_majority_vote_CV_v2(
        Xstr, ystr, groupstr,
        base_estimator, stacked_estimator,
        splitter, stacked_splitter,
        vote_type='counts', l0_pred_type='probas',
        l1_pred_type='counts',
        scale_function=None,
        stacked_scale_function=None,
        retrain_estimators=False,
        n_jobs=-1, scorer=None
        ):

    """
    Two-level VERTICAL STACKING V2

    Level 2:
        Single compound MOA classification
    Level 1:
        Strains
    Level 0:
        Bluelight conditions per strain

    Bluelight conditions --> well-level predictions

    Stacked bluelight conditions, y ---> Strain-specific well-level predictions -->
        majority_vote --> strain-specific compound-level predictions

    Vertically Stacked compound-level predictions, Y_group* --> Final compound-level predictions
    """

    l0_scores = {key: {k:{} for k in Xstr[key].keys()} for key in Xstr.keys()}
    l1_scores = {key: {} for key in Xstr.keys()}

    l1_group = []
    l1_y = []
    l1_predictions = []
    for (key1, Xs), (key1, ys), (key1, groups) in zip(Xstr.items(), ystr.items(), groupstr.items()):
        l0_y = None
        l0_predictions = []
        for (key0,X), (key0,y), (key0,group) in zip(Xs.items(),ys.items(),groups.items()):
            node = StackNode(base_estimator, splitter, is_grouped_data=True,
                             vote_type=vote_type, scale_function=scale_function,
                             n_jobs=-1, scorer=scorer)
            targ, pred = node.fit_transform(X, y, group, pred_type=l0_pred_type, group_pred=False)

            if l0_y is None:
                l0_y = targ
            else:
                assert all(l0_y == targ)
            l0_predictions.append(pred)

            l0_scores[key1][key0]['standard'] = node.scores
            l0_scores[key1][key0]['majority_vote'] = node.scores_group

        l0_predictions = np.concatenate(l0_predictions, axis=1)

        node = StackNode(stacked_estimator, splitter, is_grouped_data=True,
                         scale_function=stacked_scale_function, n_jobs=n_jobs,
                         scorer=scorer, vote_type=vote_type)
        targ, pred = node.fit_transform(l0_predictions, l0_y, group=group,
                                        pred_type=l1_pred_type, group_pred=True)

        l1_group.append(pred.index.to_list())
        l1_y.append(targ)
        l1_predictions.append(pred)

        l1_scores[key1]['standard'] = node.scores
        l1_scores[key1]['majority_vote'] = node.scores_group

    l1_predictions = np.concatenate(l1_predictions, axis=0)
    l1_y = np.concatenate(l1_y, axis=0)
    l1_group = np.concatenate(l1_group, axis=0)

    node = StackNode(stacked_estimator, stacked_splitter, is_grouped_data=True,
                    scale_function=stacked_scale_function, n_jobs=n_jobs,
                    scorer=scorer, vote_type='probas')

    node.fit(l1_predictions, l1_y, group=l1_group)

    return l0_scores, l1_scores, node.scores_group


def vertical_stacking_CV(
        Xs, ys, groups,
        base_estimator, stacked_estimator,
        splitter,
        vote_type='counts', pred_type='probas',
        scale_function=None,
        stacked_scale_function=None,
        retrain_estimators=False,
        n_jobs=-1, scorer=None,
        add_sample_weights=True
        ):

    """
    Single-level VERTICAL STACKING

    Level 1:
        Single compound MOA classification
    Level 0:
        Strains

    strains, y ---> Strain-specific well-level predictions -->
        majority_vote --> strain-specific compound-level predictions

    Vertically Stacked compound-level predictions, Y_group* --> Final compound-level predictions
    """

    l0_scores = {key: {} for key in Xs.keys()}
    l1_scores_grouped = {}
    l1_scores = {}

    l0_group = []
    l0_y = []
    l0_predictions = []
    l0g_group = []
    l0g_y = []
    l0g_predictions = []
    for (key, X), (key, y), (key, group) in zip(Xs.items(), ys.items(), groups.items()):
        node = StackNode(base_estimator, splitter, is_grouped_data=True,
                         vote_type=vote_type, scale_function=scale_function,
                         n_jobs=-1, scorer=scorer)
        node.fit(X, y, group)

        l0_scores[key]['standard'] = node.scores
        l0_scores[key]['majority_vote'] = node.scores_group

        # Get grouped predictions
        targ, pred = node.get_targets_predictions(pred_type, group_pred=True)
        l0g_group.append(pred.index.to_list())
        l0g_y.append(targ)
        l0g_predictions.append(pred)

        # Get non-grouped predictions
        targ, pred = node.get_targets_predictions(pred_type, group_pred=False)
        l0_group.append(node.group)
        l0_y.append(targ)
        l0_predictions.append(pred)

    # Use grouped predictions
    if add_sample_weights:
        sample_weight = []
        for i,key in enumerate(Xs.keys()):
            weight = np.mean(l0_scores[key]['majority_vote']['accuracy'])
            sample_weight.append(np.repeat(weight, l0g_predictions[i].shape[0]))
        sample_weight = np.concatenate(sample_weight)
    else:
        sample_weight = None

    l0g_predictions = np.concatenate(l0g_predictions, axis=0)
    l0g_y = np.concatenate(l0g_y, axis=0)
    l0g_group = np.concatenate(l0g_group, axis=0)

    node = StackNode(stacked_estimator, splitter, is_grouped_data=True,
                     scale_function=stacked_scale_function, n_jobs=n_jobs,
                     scorer=scorer, vote_type='probas')
    node.fit(l0g_predictions, l0g_y, group=l0g_group, sample_weight=sample_weight)
    l1_scores_grouped['standard'] = node.scores
    l1_scores_grouped['majority_vote'] = node.scores_group

    # Use ungrouped predictions
    if add_sample_weights:
        sample_weight = []
        for i,key in enumerate(Xs.keys()):
            weight = np.mean(l0_scores[key]['standard']['accuracy'])
            sample_weight.append(np.repeat(weight, l0_predictions[i].shape[0]))
        sample_weight = np.concatenate(sample_weight)
    else:
        sample_weight = None

    l0_predictions = np.concatenate(l0_predictions, axis=0)
    l0_y = np.concatenate(l0_y, axis=0)
    l0_group = np.concatenate(l0_group, axis=0)

    node = StackNode(stacked_estimator, splitter, is_grouped_data=True,
                     scale_function=stacked_scale_function, n_jobs=n_jobs,
                     scorer=scorer, vote_type=vote_type)
    node.fit(l0_predictions, l0_y, group=l0_group, sample_weight=sample_weight)
    l1_scores['standard'] = node.scores
    l1_scores['majority_vote'] = node.scores_group

    return l0_scores, l1_scores, l1_scores_grouped


def strain_stack_per_dose_CV(
        Xs, ys, groups, doses, blues,
        base_estimator, stacked_estimator,
        splitter, drug_moa_mapper,
        vote_type='counts', pred_type='probas',
        scale_function=None,
        stacked_scale_function=None,
        retrain_estimators=False,
        n_jobs=-1, scorer=None,
        add_bluelight_id_to_stacked_pred=False
        ):

    """
    Single-level VERTICAL STACKING

    Level 1:
        Single compound MOA classification
    Level 0:
        Strains

    strains, y ---> Strain-specific well-level predictions -->
        majority_vote --> strain-specific compound-level predictions

    Vertically Stacked compound-level predictions, Y_group* --> Final compound-level predictions
    """
    from tierpsytools.analysis.drug_screenings.bagging_drug_data import StrainAugmentDrugData
    from sklearn.preprocessing import OneHotEncoder

    l0_scores = {key: {} for key in Xs.keys()}
    l1_scores = {}

    l0_pred = {}
    for (key, X), (key, y), (key, group) in zip(Xs.items(), ys.items(), groups.items()):
        node = StackNode(base_estimator, splitter, is_grouped_data=True,
                         vote_type=vote_type, scale_function=scale_function,
                         n_jobs=n_jobs, scorer=scorer)
        node.fit(X, y, group)

        l0_scores[key]['standard'] = node.scores
        l0_scores[key]['majority_vote'] = node.scores_group

        # Get non-grouped predictions
        targ, pred = node.get_targets_predictions(pred_type, group_pred=False)
        assert all(node.y == ys[key])
        assert all(node.group == groups[key])
        l0_pred[key] = pd.DataFrame(pred, columns=node.labels)

    # Concatenate predictions
    augmenter = StrainAugmentDrugData(
        n_augmented_bags=1, replace=False, frac_per_dose=1, random_state=None,
        bluelight_conditions=True)
    X_aug, group_aug, dose_aug, blue_aug = augmenter.fit_transform(
        l0_pred, groups, doses, blues, shuffle=True)

    # Add blue_id feature
    if add_bluelight_id_to_stacked_pred:
        enc = OneHotEncoder(sparse=False)
        blue_ids = enc.fit_transform(blue_aug.reshape(-1,1))
        blue_ids = pd.DataFrame(blue_ids, columns=enc.get_feature_names(input_features=['bluelight']))
        X_aug = pd.concat([X_aug,blue_ids], axis=1)

    # drop samples with nans
    keep_ids = ~X_aug.isna().any(axis=1).values

    # Get classes
    y_aug = pd.Series(group_aug).map(drug_moa_mapper).values

    node = StackNode(stacked_estimator, splitter, is_grouped_data=True,
                     scale_function=stacked_scale_function, n_jobs=n_jobs,
                     scorer=scorer, vote_type='probas')
    node.fit(X_aug[keep_ids], y_aug[keep_ids], group=group_aug[keep_ids])
    l1_scores['standard'] = node.scores
    l1_scores['majority_vote'] = node.scores_group

    return l0_scores, l1_scores



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
    for key in Xs.keys():
        X = np.array(Xs[key])
        y = np.array(ys[key])
        group = np.array(groups[key])

        _scores, _scores_maj, pred, probas, labels = majority_vote_CV(
            X, y, group, base_estimator, splitter,
            scale_function=scale_function, n_jobs=n_jobs, vote_type=vote_type,
            return_predictions=True, scorer=scorer
            )

        scores[key] = _scores
        scores_group[key] = _scores_maj

        # Majority vote
        if pred_type=='single_pred':
            pred_votes[key] = get_majority_vote(group, labels, y_pred=pred, probas=probas)
        else:
            pred_votes[key] = get_sum_of_votes(group, labels, y_pred=pred, probas=probas, sum_type=pred_type)

    ## Stack
    #--------
    for key in pred_votes.keys():
        pred_votes[key] = pred_votes[key].rename(
            columns={col:'_'.join([str(col),key]) for col in pred_votes[key].columns})

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
        for key,X in Xs.items():
            base_estimator.fit(X,y)
            trained_estimators[key] = base_estimator
        # stacked
        stacked_estimator.fit(pred_votes, y_group)
        trained_estimators['stacked'] = stacked_estimator

    return scores, scores_group, trained_estimators


