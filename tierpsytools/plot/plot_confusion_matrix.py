#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:49:02 2020

@author: em812
"""

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
    rc_params = {
            'font.sans-serif': "Arial",
            'svg.fonttype': 'none',
            }

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
        with plt.rc_context(rc_params):
            plt.savefig(saveto)
        plt.close()
    return
