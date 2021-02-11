#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similar CV wrappers exist in sklearn. The ones developed here:
    - allow easier scaling (without using pipeline)
    - can use custom splitters for grouped data
    - have customized output that are convenient for the analysis of the
    drug screenings (for example in majority vote functions).

Created on Thu Apr 23 19:33:54 2020

@author: em812
"""
import numpy as np
from tierpsytools.preprocessing.scaling_class import scalingClass
from tierpsytools.analysis.helper import _get_multi_sclassifscorers
from joblib import Parallel, delayed
import pdb

def cv_predict(
        X, y,
        splitter, estimator,
        group=None, scale_function=None,
        n_jobs=-1, sample_weight=None
        ):

    if n_jobs==1:
        return cv_predict_single(X, y, splitter, estimator, group=group,
                                 scale_function=scale_function,
                                 sample_weight=None)

    def _one_fit(X, y, group, train_index, test_index, estimator,
                 sample_weight, scale_function=None):
        # Normalize
        scaler = scalingClass(scaling=scale_function)
        X_train = scaler.fit_transform(X[train_index])
        X_test = scaler.transform(X[test_index])

        # Train classifier
        if sample_weight is None:
            estimator.fit(X_train, y[train_index])
        else:
            estimator.fit(X_train, y[train_index], sample_weight=sample_weight[train_index])

        # Predict
        y_pred = estimator.predict(X_test)
        assert all(estimator.classes_ == np.unique(y)), \
            'Not all classes are represented in the folds.'
        if hasattr(estimator, 'predict_proba'):
            y_probas = estimator.predict_proba(X_test)
        elif hasattr(estimator, 'decision_function'):
            y_probas = estimator.decision_function(X_test)
        elif hasattr(estimator, 'oob_decision_function') and estimator.oob_score:
            y_probas = estimator.oob_decision_function(X_test)
        else:
            y_probas = None
        return test_index, y_pred, y_probas, estimator

    X = np.array(X)
    y = np.array(y)
    labels = np.unique(y)

    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    pred = np.empty_like(y)
    probas = np.empty((X.shape[0], labels.shape[0]))

    parallel = Parallel(n_jobs=n_jobs, verbose=True)
    func = delayed(_one_fit)

    res = parallel(
        func(X, y, group, train_index, test_index, estimator,
             sample_weight, scale_function=scale_function)
        for train_index, test_index in splitter.split(X, y, group))

    for  test_index,y_pred,y_probas,_ in res:
        pred[test_index] = y_pred
        if y_probas is not None:
            probas[test_index] = y_probas
    test_folds = [test_index for test_index,_,_,_ in res]
    trained_estimators = [est for _,_,_,est in res]

    return pred, probas, labels, test_folds, trained_estimators

def cv_predict_single(
        X, y,
        splitter, estimator,
        group=None, scale_function=None,
        sample_weight=None
        ):

    labels = np.unique(y)
    X = np.array(X)
    y = np.array(y)

    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    pred = np.empty_like(y)
    probas = np.empty((X.shape[0], labels.shape[0]))

    test_folds = []
    trained_estimators = []
    for train_index, test_index in splitter.split(X, y, group):

        test_folds.append(test_index)

        # Normalize
        scaler = scalingClass(scaling=scale_function)
        X_train = scaler.fit_transform(X[train_index])
        X_test = scaler.transform(X[test_index])

        # Train classifier
        if sample_weight is not None:
            estimator.fit(X_train, y[train_index], sample_weight[train_index])
        else:
            estimator.fit(X_train, y[train_index])
        trained_estimators.append(estimator)

        # Predict
        pred[test_index] = estimator.predict(X_test)
        assert all(estimator.classes_ == labels), \
            'Not all classes are represented in the folds.'

        if hasattr(estimator, 'predict_proba'):
            probas[test_index] = estimator.predict_proba(X_test)
        elif hasattr(estimator, 'decision_function'):
            probas[test_index] = estimator.decision_function(X_test)
        elif hasattr(estimator, 'oob_decision_function') and estimator.oob_score:
            probas[test_index] = estimator.oob_decision_function(X_test)
        else:
            probas = None

    return pred, probas, labels, test_folds, trained_estimators


def cv_score(
        X, y,
        splitter, estimator,
        group=None, sample_weight=None,
        scale_function=None,
        n_jobs=-1, scorer=None,
        return_predictions=False):

    scorers = _get_multi_sclassifscorers(scorer)

    if n_jobs==1:
        pred, probas, labels, test_folds, _ = \
            cv_predict(X, y, splitter, estimator, group=group,
                       scale_function=scale_function)
    else:
        pred, probas, labels, test_folds, _ = \
            cv_predict(X, y, splitter, estimator, group=group,
                       scale_function=scale_function, n_jobs=n_jobs,
                       sample_weight=sample_weight)

    scores = {key:[] for key in scorers.keys()}
    for test_index in test_folds:
        if probas is not None:
            _probas = probas[test_index]
        else:
            _probas = None
        for key in scorers.keys():
            scores[key].append(scorers[key].score(
                y[test_index], pred=pred[test_index], probas=_probas,
                labels=labels, sample_weight=sample_weight)
                )

    if return_predictions:
        return scores, pred, probas, labels
    else:
        return scores


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

def rearrange_confusion_matrix(cm, n_clusters):
    from sklearn.cluster import SpectralCoclustering

    clst = SpectralCoclustering(n_clusters=n_clusters).fit(cm)

    idx = []
    for c in range(n_clusters):
        idx.append(clst.get_indices(c)[0])
    idx = np.concatenate(idx)

    cm_clustered = np.zeros(cm.shape, dtype=int)

    for i, idxi in enumerate(idx):
        for j, idxj in enumerate(idx):
            cm_clustered[i,j] = cm[idxi, idxj]

    return cm_clustered, idx

def plot_confusion_matrix(
        y_true, y_pred, classes=None, normalize=False, title=None, figsize=(8,8),
        cmap=None, saveto=None, cluster=False, n_clusters=3, add_colorbar=False,
        show_labels=True, show_counts=True
        ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = 'Arial'

#    if not title:
#        if normalize:
#            title = 'Normalized confusion matrix'
#        else:
#            title = 'Confusion matrix, without normalization'

    if cmap is None:
        cmap = plt.cm.Blues

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    if classes is not None:
        classes = np.array([classes[key] for key in unique_labels(y_true, y_pred)])
    else:
        classes = unique_labels(y_true, y_pred)

    if cluster:
        try:
            cm, idx = rearrange_confusion_matrix(cm, n_clusters)
            classes = classes[idx]
        except:
            print('Waring: The confusion matrix could not be clustered.')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    if add_colorbar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(im, cax=cax)

    if show_labels:
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    if show_counts:
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if saveto is not None:
        plt.savefig(saveto)
        plt.close()
    return
