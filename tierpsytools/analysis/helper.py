#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:12:26 2020

@author: em812
"""

import numpy as np

def _check_if_classifscorer(scorer):
    from inspect import isfunction
    from tierpsytools.analysis.scorers import ClassifScorer

    if scorer is None:
        scorer = ClassifScorer()
    elif isfunction(scorer):
        raise Exception('The scorer input must be a ClassifScorer object. '+
                        'Define a ClassifScorer object with the scorer '+
                        'function.')
    elif not isinstance(scorer, ClassifScorer):
        raise Exception('scorer input not recognised. The scorer must be a '+
                        'ClassifScorer object or None.')
    return scorer

def _get_pred_from_probas(probas, labels):
    pred = [labels[maxind] for maxind in np.argmax(probas, axis=1)]
    pred = np.array(pred)
    return pred