#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:33:54 2020

@author: em812
"""

def RFE_selection(X, y, n_feat, estimator, step=100, save_to=None):
    from sklearn.feature_selection import RFE
    from time import time
    import pickle

    print('RFE selection for n_feat={}.'.format(n_feat))
    start_time = time()

    rfe = RFE(estimator, n_features_to_select=n_feat, step=step)
    X_sel = rfe.fit_transform(X,y)

    print("RFE: --- %s seconds ---" % (time() - start_time))

    if save_to is not None:
        pickle.dump( rfe, open(save_to/'fitted_rfe_nfeat={}.p'.format(n_feat), "wb") )

    return X_sel, rfe.support_, rfe

def kbest_selection(X, y, n_feat, score_func=None):
    from sklearn.feature_selection import SelectKBest, f_classif

    if score_func is None:
        score_func = f_classif

    selector = SelectKBest(score_func=score_func, k=n_feat)
    X_sel = selector.fit_transform(X, y)

    return X_sel, selector.support_


def model_selection(X, y, estimator, param_grid, cv_strategy=0.2, save_to=None, saveid=None):
    from sklearn.model_selection import GridSearchCV
    from time import time
    import pickle

    print('Starting grid search CV...')
    start_time = time()
    grid_search = GridSearchCV(
        estimator, param_grid=param_grid, cv=cv_strategy, n_jobs=-1, return_train_score=True)

    grid_search.fit(X, y)
    print("Grid search: --- %s seconds ---" % (time() - start_time))

    if save_to is not None:
        pickle.dump( grid_search, open(save_to/'fitted_gridsearchcv_nfeat={}.p'.format(saveid), "wb") )

    return grid_search.best_estimator_, grid_search.best_score_

def plot_confusion_matrix(
        y_true, y_pred, classes=None, normalize=False, title=None, figsize=(8,8),
        cmap=None, saveto=None
        ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np

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
        classes = [classes[key] for key in unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.figure.colorbar(im, cax=cax)
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

def get_fscore(
        y_true, y_pred, classes=None,
        output_precision_recall=True,
        plot_confusion=False, saveto=None):
    """
    Estimate the precision, recall and fscore for a multiclass classification problem
    param:
        y_true: the true class labels of the samples (array size n_samples)
        y_pred: the predicted class labels from the classfier (array size n_samples)
        plot: boolean defining whether precision,recall and f_score will be ploted
    return:
        precision: the precision score of each class (array size n_classes)
        recall: the recall score of each class (array size n_classes)
        fscore: the f1-score of each class (array size n_classes)
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

    if output_precision_recall:
        return precision,recall,fscore
    else:
        return fscore





