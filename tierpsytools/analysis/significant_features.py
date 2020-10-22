#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:39:29 2019

@author: em812
"""
import pdb
from tierpsytools.feature_processing.scaling_class import scalingClass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_boxplots(
        feat_to_plot, y_class, scores, feat_df, pvalues=None,
        figsize=None, saveto=None, xlabel=None,
        close_after_plotting=False
        ):

    classes = np.unique(y_class)
    for i,ft in enumerate(feat_to_plot):
        title = ft+'\n'+'score={:.3f}'.format(scores[i])
        if pvalues is not None:
            title+=' - p-value = {}'.format(pvalues[i])
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.boxplot([feat_df.loc[y_class==cl,ft] for cl in classes])
        plt.xticks(list(range(1,classes.shape[0]+1)), classes)
        plt.ylabel(ft)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if saveto:
            plt.savefig(saveto/ft)
        if close_after_plotting:
            plt.close()
    return

def k_significant_feat(
        feat, y_class, k=5, score_func='f_classif', scale=None,
        feat_names=None, plot=True, k_to_plot=None, close_after_plotting=False,
        saveto=None, figsize=None, title=None, xlabel=None
        ):
    """
    Finds the k most significant features in the feature matrix, based on
    how well they separate the data in groups defined in y_class. It uses
    univariate statistical tests (the type of test is specified in the variable
    score_func).
    param:
        feat: array-like, shape=(n_samlples, n_features)
            The feature matrix
        y_class: array-like, shape=(n_samples)
            Vector with the class of each samples
        k: integer or 'all'
            Number of fetures to select
        score_func: str or function, optional
            If string 'f_classif', 'chi2', 'mutual_info_classif' then the
            function f_classif, chi2 or mutual_info_classif
            from sklearn.feature_selection will be used.
            Otherwise, the user needs to input a function that takes two
            arrays X and y, and returns a pair of arrays (scores, pvalues)
            or a single array with scores.
            Default is 'f_classif'.
        scale: None, str or function, optional
            If string 'standardize', 'minmax_scale', the
            tierpsytools.feature_processing.scaling_class.scalingClass is used
            to scale the features.
            Otherwise the used can input a function that scales features.
            Default is None (no scaling).
        feat_names: list shape=(n_features)
            The names of the features, when feat is an array and not a dataframe
            (will be used for plotting)

    return:
        support: array of booleans
            True for the selected features, False for the rest
        plot: boolean
            If True, the boxplots of the chosen features will be plotted
        plot
    """
    from sklearn.feature_selection import \
        SelectKBest, chi2,f_classif, mutual_info_classif

    if plot and k_to_plot is None:
        k_to_plot = k

    if isinstance(feat,np.ndarray):
        feat = pd.DataFrame(feat, columns=feat_names)
    feat = feat.loc[:, feat.std()!=0]

    if isinstance(k,str):
        if k=='all':
            k = feat.shape[1]

    # Find most significant features
    if isinstance(score_func, str):
        if score_func=='f_classif':
            score_func = f_classif
        elif score_func=='chi2':
            score_func = chi2
        elif score_func=='mutual_info_classif':
            score_func = mutual_info_classif

    if scale is not None:
        if isinstance(scale, str):
            scaler = scalingClass(scaling=scale)
            feat_scaled = scaler.fit_transform(feat)
        else:
            feat_scaled = scale(feat)
    else:
        feat_scaled = feat

    skb = SelectKBest(score_func=score_func, k=k)
    skb.fit(feat_scaled, y_class)

    support = skb.get_support()
    top_ft_ids = np.flip(np.argsort(skb.scores_))[:k]
    scores = skb.scores_[top_ft_ids]
    if hasattr(skb, 'pvalues_'):
        pvalues = skb.pvalues_[top_ft_ids]
    else:
        pvalues = None

    # Plot a boxplot for each feature, showing its distribution in each class
    if plot:
        plot_feature_boxplots(
            feat.iloc[:, top_ft_ids[:k_to_plot]], y_class, scores, feat,
            pvalues=pvalues, figsize=figsize, saveto=saveto, xlabel=xlabel,
            close_after_plotting=close_after_plotting)

    if pvalues is not None:
        return feat.columns[top_ft_ids].to_list(), (scores, pvalues), support
    else:
        return feat.columns[top_ft_ids].to_list(), scores, support



def top_feat_in_PCs(X, pc=0, scale=False, k='auto', feat_names=None):
    """
    Runs PCA and gives the top k contributing features for the specified
    principal component (by default the first component).

    """
    import pandas as pd
    from sklearn.decomposition import PCA

    if isinstance(X,np.ndarray):
        X = pd.DataFrame(X, columns=feat_names)

    if scale:
        Xscaled = X.loc[:,X.std()!=0].copy()
        Xscaled = ( Xscaled - Xscaled.mean() ) / Xscaled.std()
    else:
        Xscaled = X

    pca = PCA(n_components=pc+1)
    pca.fit(Xscaled)

    ## pca.components_ --> each row contains the feature coefficients for one component
    component = pca.components_[pc,:]

    sortid = np.flip(np.argsort(np.abs(component)))

    if k=='auto':
        from kneed import KneeLocator
        kn = KneeLocator(np.arange(sortid.shape[0]), np.abs(component[sortid]), curve='convex', direction='decreasing')
        k = kn.knee

    k_feat = list(Xscaled.columns[sortid[:k]])

    return k_feat, component[sortid[:k]]


def top_feat_in_LDA(
        X, y, ldc=[0,1],
        scale=False, k='auto',
        feat_names=None, estimator=None
        ):
    """
    Runs LDA and gives the top k contributing features for the specified
    linear discriminant component (by default the first component).

    """
    import pandas as pd
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if isinstance(X,np.ndarray):
        X = pd.DataFrame(X, columns=feat_names)

    if scale:
        Xscaled = X.loc[:,X.std()!=0].copy()
        Xscaled = ( Xscaled - Xscaled.mean() ) / Xscaled.std()
    else:
        Xscaled = X

    if estimator is None:
        estimator = LinearDiscriminantAnalysis(n_components=max(ldc)+1)

    estimator.fit(Xscaled, y)

    ## pca.components_ --> each row contains the feature coefficients for one component
    component = estimator.scalings_[:, ldc]
    if isinstance(ldc,int):
        importance = np.abs(component)
    else:
        importance = np.linalg.norm(component, axis=1, ord=1)

    sortid = np.flip(np.argsort(importance))

    if k=='auto':
        from kneed import KneeLocator
        kn = KneeLocator(np.arange(sortid.shape[0]), importance[sortid], curve='convex', direction='decreasing')
        k = kn.knee

    k_feat = list(Xscaled.columns[sortid[:k]])

    return k_feat, importance[sortid[:k]]


def k_significant_from_classifier(
        feat, y_class, estimator, k=5, scale=None,
        feat_names=None, k_to_plot=None, close_after_plotting=False,
        saveto=None, figsize=None, title=None, xlabel=None
        ):
    """
    param:
        feat: array-like, shape=(n_samlples, n_features)
            The feature matrix
        y_class: array-like, shape=(n_samples)
            Vector with the class of each samples
        estimator: object
            A supervised learning estimator with a fit method that provides
            information about feature importance either through a coef_
            attribute or through a feature_importances_ attribute.
        k: integer or 'all', optional
            Number of fetures to select
            Default is 5.
        scale: None, str or function, optional
            If string 'standardize', 'minmax_scale', the
            tierpsytools.feature_processing.scaling_class.scalingClass is used
            to scale the features.
            Otherwise the used can input a function that scales features.
            Default is None (no scaling).
        feat_names: list shape=(n_features)
            The names of the features, when feat is an array and not a dataframe
            (will be used for plotting)
        plot: boolean
            If True, the boxplots of the chosen features will be plotted

    return:
        top_feat: list, shape=(k,)
            The names or indexes (if feature names are not given) of the top features,
            sorted by importance.
        scores: array-like, shape=(k,)
            The scores of the top features, sorted by importance.
        support: array of booleans, shape=(n_features,)
            True for the selected features, False for the rest
        plot
    """
    from tierpsytools.feature_processing.scaling_class import scalingClass

    if plot and k_to_plot is None:
        k_to_plot = k

    if isinstance(feat,np.ndarray):
        feat = pd.DataFrame(feat,columns=feat_names)

    if isinstance(k,str):
        if k=='all':
            k = feat.shape[1]

    if scale is not None:
        if isinstance(scale, str):
            scaler = scalingClass(scaling=scale)
            feat_scaled = scaler.fit_transform(feat)
        else:
            feat_scaled = scale(feat)
    else:
        feat_scaled = feat

    estimator.fit(feat_scaled, y_class)
    if hasattr(estimator, 'coef_'):
        scores = np.linalg.norm(estimator.coef_, axis=0, ord=1)
    elif hasattr(estimator, 'feture_importances_'):
        scores = estimator.feture_importances_
    else:
        raise ValueError('The chosen estimator does not have a coef_ attribute'+
                         ' or a feature_importances_ attribute.')

    top_ft_ids = np.flip(np.argsort(scores))[:k]
    support = np.zeros(feat.shape[1]).astype(bool)
    support[top_ft_ids] = True
    scores = scores[top_ft_ids]

    # Plot a boxplot for each feature, showing its distribution in each class
    if plot:
        plot_feature_boxplots(
            feat.iloc[:, top_ft_ids[:k_to_plot]], y_class, scores,
            figsize=figsize, saveto=saveto, xlabel=xlabel,
            close_after_plotting=close_after_plotting)

    top_feat = feat.columns[top_ft_ids].to_list()
    return top_feat, scores, support


if __name__=="__main__":
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    import numpy as np

    X1 = np.concatenate(
        [np.random.normal(loc=0.0, scale=1, size=100),
         np.random.normal(loc=2.0, scale=1, size=100)]).reshape(-1,1)
    X2 = np.concatenate(
        [np.random.normal(loc=0.0, scale=1, size=100),
         np.random.normal(loc=1.0, scale=1, size=100)]).reshape(-1,1)
    X3 = np.concatenate(
        [np.random.normal(loc=0.0, scale=2, size=100),
         np.random.normal(loc=1.0, scale=2, size=100)]).reshape(-1,1)
    X4 = np.random.rand(200).reshape(-1,1)
    X5 = np.random.rand(200).reshape(-1,1)

    X = np.concatenate([X1,X2,X3,X4,X5], axis=1)

    y = np.concatenate([np.zeros(100), np.ones(100)])

    top_feat1, scores1, support1 = k_significant_feat(
        X, y, k=5, score_func='f_classif', scale='standardize',
        feat_names=['X1','X2','X3','X4','X5'], figsize=None,
        title=None, savefig=None, close_after_plotting=False)

    estimator = LogisticRegression(penalty='l2', C=100) #SVC(kernel='linear')

    top_feat2, scores2, support2 = k_significant_from_classifier(
        X, y, estimator, k=5, scale=None,
        feat_names=['X1','X2','X3','X4','X5'], figsize=None, title=None, savefig=None,
        close_after_plotting=False)

    top_feat3, scores3 = top_feat_in_PCs(X, pc=0, scale=True, k=5, feat_names=['X1','X2','X3','X4','X5'])
