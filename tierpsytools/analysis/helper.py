#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:12:26 2020

@author: em812
"""

import numpy as np
from inspect import isfunction, isclass

def _get_multi_sclassifscorers(scorer):
    if not isinstance(scorer, list):
        scorer = [scorer]

    score_dict = dict()

    for score in scorer:
        scorer_obj =  _get_classifscorer(score)
        score_dict[scorer_obj.name] = scorer_obj

    return score_dict

def _get_classifscorer(scorer):
    from tierpsytools.analysis.scorers import ClassifScorer

    if scorer is None:
        scorer = ClassifScorer(scorer='accuracy')
    elif isfunction(scorer):
        raise Exception('The scorer input must be a ClassifScorer object. '+
                        'Define a ClassifScorer object with the scorer '+
                        'function.')
    elif isinstance(scorer, str):
        scorer = ClassifScorer(scorer=scorer)
    elif not isclass(type(ClassifScorer())):
        raise Exception('scorer input not recognised. The scorer must be a '+
                        'ClassifScorer object or None.')
    return scorer

def _get_pred_from_probas(probas, labels):
    pred = [labels[maxind] for maxind in np.argmax(probas, axis=1)]
    pred = np.array(pred)
    return pred