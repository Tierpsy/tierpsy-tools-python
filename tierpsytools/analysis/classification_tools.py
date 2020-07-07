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
