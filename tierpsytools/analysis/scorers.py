#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:15:37 2020

@author: em812
"""
from sklearn.metrics import *
from tierpsytools.analysis.drug_screenings.MIL.majority_vote import \
    _get_y_group, get_majority_vote, get_sum_of_votes
from tierpsytools.analysis.helper import _get_pred_from_probas
import pandas as pd
import warnings
import pdb


class ClassifScorer():
    def __init__(self, scorer=None, name=None, require_proba=False,
                 accept_weights=False, **kwargs):

        if scorer is None:
            scorer = 'accuracy'

        if isinstance(scorer,str):
            self.scorer, self.require_proba, self.accept_weights = \
                _make_scorer(scorer, **kwargs)
            self.name = scorer
        else:
            self.scorer = scorer
            self.name = name
            self.require_proba = require_proba
            self.accept_weights = accept_weights


    def _get_score(self, true, pred, probas, sample_weight):

        if self.require_proba:
            if self.accept_weights:
                _score = self.scorer(true, probas, sample_weight=sample_weight)
            else:
                _score = self.scorer(true, probas)
        else:
            if self.accept_weights:
                _score = self.scorer(true, pred, sample_weight=sample_weight)
            else:
                _score = self.scorer(true, pred)

        return _score

    def _check_input(self, true, pred, probas, labels, sample_weight, *argv):

        if pred is None and probas is None:
            raise ValueError('Must give either pred or probas to the score function.')
        if probas is not None and labels is None:
            raise ValueError('Must give class labels with probabilities.')
        if self.require_proba and probas is None:
            raise ValueError('Scorer requires proba predictions.')
        if not self.accept_weights:
            if sample_weight is not None:
                warnings.warn('Scorer does not accept sample weights. Weights will be ignored.')

        if argv:
            vote_type = argv[0]
            if vote_type == 'counts' and self.require_proba:
                warnings.warn('The scoring function requires probabilities.' +
                              'The vote_type input will be ignored.')
            if vote_type is None and not self.require_proba:
                raise ValueError('Must define vote_type for the majority vote predictions.')
        return

    def score(self, true,
              pred=None, probas=None,
              labels=None, sample_weight=None
              ):

        self._check_input(true, pred, probas, labels, sample_weight)

        if not self.require_proba and pred is None:
            pred = _get_pred_from_probas(probas, labels)

        return self._get_score(true, pred, probas, sample_weight)

    def score_maj(self, true, group,
                  pred=None, probas=None, labels=None,
                  sample_weight=None, vote_type=None):

        self._check_input(true, pred, probas, labels, sample_weight, vote_type)

        if not self.require_proba and pred is None:
            pred = _get_pred_from_probas(probas, labels)

        y_true = _get_y_group(true, group)
        if sample_weight is not None:
            y_weight = pd.DataFrame(sample_weight).groupby(by=group).mean()
            y_weight = y_weight.loc[y_true.index, :].values.reshape(-1)
        else:
            y_weight=None
        if self.require_proba:
            y_pred = None
            y_probas = get_sum_of_votes(
                group, labels, y_pred=pred, probas=probas, sum_type='probas',
                normalized=True)
            y_probas = y_probas.loc[y_true.index, :].values
        else:
            y_pred = get_majority_vote(
                group, y_pred=pred, probas=probas, labels=labels,
                vote_type=vote_type)
            y_pred = y_pred.loc[y_true.index].values
            y_probas = None
        y_true = y_true.values

        return self._get_score(y_true, y_pred, y_probas, y_weight)


def _make_scorer(scorer, **kwargs):
    from functools import partial

    SCORERS = dict(
        accuracy = {'scoring': accuracy_score, 'probas':False, 'weights':True},
        accuracy_score = {'scoring': accuracy_score, 'probas':False, 'weights':True},
        balanced_accuracy_score = {'scoring': balanced_accuracy_score, 'probas':False, 'weights':True},
        balanced_accuracy = {'scoring': balanced_accuracy_score, 'probas':False, 'weights':True},
        f1 = {'scoring': f1_score, 'probas':False, 'weights':True},
        f1_score = {'scoring': f1_score, 'probas':False, 'weights':True},
        roc_auc = {'scoring': roc_auc_score, 'probas':True, 'weights':False},
        roc_auc_score = {'scoring': roc_auc_score, 'probas':True, 'weights':False},
        jaccard = {'scoring': jaccard_score, 'probas':False, 'weights':True},
        jaccard_score = {'scoring': jaccard_score, 'probas':False, 'weights':True},
        log_loss = {'scoring': log_loss, 'probas':True, 'weights':True},
        hinge_loss = {'scoring': hinge_loss, 'probas':True, 'weights':True},
        hamming_loss = {'scoring': hamming_loss, 'probas':False, 'weights':True},
        cohen_kappa = {'scoring': cohen_kappa_score, 'probas':False, 'weights':True},
        cohen_kappa_score = {'scoring': cohen_kappa_score, 'probas':False, 'weights':True},
        mcc = {'scoring': matthews_corrcoef, 'probas':False, 'weights':True},
        matthews_corrcoef = {'scoring': matthews_corrcoef, 'probas':False, 'weights':True},
        precision = {'scoring': precision_score, 'probas':False, 'weights':True},
        precision_score = {'scoring': precision_score, 'probas':False, 'weights':True},
        recall = {'scoring': recall_score, 'probas':False, 'weights':True},
        recall_score = {'scoring': recall_score, 'probas':False, 'weights':True},
        )

    return partial(SCORERS[scorer]['scoring'], **kwargs), \
        SCORERS[scorer]['probas'], SCORERS[scorer]['weights']