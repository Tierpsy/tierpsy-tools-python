#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:33:54 2020

@author: em812
"""

def get_fscore(
        y_true, y_pred,
        return_precision_recall=True):
    """
    Estimate the precision, recall and fscore for a multiclass classification problem
    param:
        y_true: the true class labels of the samples (array size n_samples)
        y_pred: the predicted class labels from the classfier (array size n_samples)
        return_precision_recall: boolean defining whether precision and recall
        will be returned together with the f_score
    return:
        fscore: the f1-score of each class (array size n_classes)
        precision (optional): the precision score of each class (array size n_classes)
        recall (optional): the recall score of each class (array size n_classes)
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np

    confMat =  confusion_matrix(y_true, y_pred, labels=np.unique(y_true), sample_weight=None)

    precision = np.empty(confMat.shape[0])
    recall = np.empty(confMat.shape[0])
    fscore = np.empty(confMat.shape[0])
    for i in range(confMat.shape[0]):
        precision[i] = confMat[i,i]/np.sum(confMat[i,:])
        recall[i] = confMat[i,i]/np.sum(confMat[:,i])
        fscore[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    if return_precision_recall:
        return fscore, precision, recall
    else:
        return fscore





