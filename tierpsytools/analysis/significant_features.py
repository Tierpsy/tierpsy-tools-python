#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:39:29 2019

@author: em812
"""
import pdb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tierpsytools.analysis.significant_features_helper import plot_feature_boxplots
from tierpsytools.preprocessing.scaling_class import scalingClass

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
            tierpsytools.preprocessing.scaling_class.scalingClass is used
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
            feat.iloc[:, top_ft_ids[:k_to_plot]], y_class, scores,
            pvalues=pvalues, figsize=figsize, saveto=saveto, xlabel=xlabel,
            close_after_plotting=close_after_plotting)

    if pvalues is not None:
        return feat.columns[top_ft_ids].to_list(), (scores, pvalues), support
    else:
        return feat.columns[top_ft_ids].to_list(), scores, support


def mRMR_feature_selection(
        feat, k=10, y_class=None, redundancy=None, relevance=None,
        redundancy_func='pearson_corr', get_abs_redun=False,
        relevance_func='kruskal',
        n_bins=10, mrmr_criterion='MIQ',
        normalize_redundancy=False, normalize_relevance=True,
        plot=True, k_to_plot=None, close_after_plotting=False,
        saveto=None, figsize=None
        ):
    """
    Finds k features with the best mRMR score (minimum Redunduncy, Maximum Relevance).
    The redundancy and relevance can be passed to the function as pre-calculated
    measures. If they are not precalculated, then they can be calculated using
    multual information or another criterion from the available options.
    param:
        feat: array-like, shape=(n_samlples, n_features)
            The feature matrix
        y_class: array-like, shape=(n_samples)
            Vector with the class of each sample. If the reduncdancy and relevance
            measures are precacaluclated, this parameter is ignored.
        k: integer or 'all'
            Number of fetures to select. If all, all the features will be ranked
            and the mRMR scores for every feature will be calculated. The 'all'
            option is not recommended when the number of candidate features is large.
        redundancy: array-like shape=(n_features, n_features) or None
            pre-computed correlation-like measure that will be used as the
            redundancy component of the mRMR score.
            Attention: the values must be normalized to match the relevance
            component.
            If None, the redundancy will be calculated based on the redundancy_func.
        relevance: array-like, shape=(n_features,) or None
            Pre-computed significance measure for each candidate feature
            that will be used as the relevance component of the mRMR score.
            Attention: the values must be normalized to match the redundancy
            component.
            If None, the relevance will be calculated based on the relevance_func
            using the y_class information.
        redundancy_func: str or function, optional
            The function to use to calculate the redundancy component of the mRMR
            score of a feature with respect to a feature set.
            If string, then one can choose between the options:
                'mutual_info', 'pearson_corr', 'kendall_corr' and 'spearman_corr'.
            Otherwise, the user needs to input a function that takes two
            arrays and returns a score.
            or a single array with scores.
            Default is 'pearson_corr'.
            Attention: when mutual_info is used, the features are discretized
            in bins. The number of bins can be customized using the parameter
            n_bins.
        get_abs_redun : bool, default is False
            This parameter is used only when a function instance is passed as
            redundancy_func or when pre-computed redundancy measures are passed.
            When redundancy_func is a string, then this parameter is ignored.
            With this parameter, the user can choose whether to get the abs
            values of the redundancy scores before summing up to get the redundancy
            component of the mrmr score. For example, if we use a correlation-like
            measure, low negative values  (anticorrelation) signify high redundancy
            similar to high positive  values. In this case, abs values must be used
            for the mrmr score.
        relevance_func: str or function, optional
            If string 'kruskal', 'f_classif', 'chi2', 'mutual_info' then the
            function f_classif, chi2 or mutual_info_classif
            from sklearn.feature_selection will be used.
            Otherwise, the user needs to input a function that takes a feature
            array x and the class labels y, and returns a score and a pvalue
            or a single score.
            Default is 'kruskal'.
        n_bins: int, default is 5
            number of bins to use to discretize the features to calculate the
            mutual_info.
    return:
        list of top ranked features
        mrmr_scores : array, shape=(k,)
            The mrmr_scores of the selected features.
        support : array of booleans
            True for the selected features, False for the rest
    """
    from tierpsytools.analysis.significant_features_helper import get_redundancy, get_relevance, mRMR_select

    k_type_error = "Data type of k not recognized."

    if isinstance(k, str):
        if k == 'all':
            k = feat.shape[1]
        else:
            print(k_type_error)
    elif not isinstance(k, int):
        print(k_type_error)

    if redundancy is None:
        redundancy = get_redundancy(feat, redundancy_func, get_abs_redun, n_bins)
    else:
        if get_abs_redun:
            redundancy = np.abs(redundancy)

    if relevance is None:
        relevance = get_relevance(feat, y_class, relevance_func)

    if normalize_redundancy:
        redundancy = redundancy/np.max(redundancy)
    if normalize_relevance:
        relevance = relevance/np.max(relevance)

    # Initialize selected feature set with the feature with max relevance
    top_ft_ids = np.argmax(relevance)
    mrmr_scores = np.inf

    # Add with forward selection
    for i in range(k-1):
        top_ft_ids, score = mRMR_select(
            top_ft_ids, redundancy, relevance, criterion=mrmr_criterion)
        mrmr_scores = np.append(mrmr_scores, score)

    support = np.zeros(feat.shape[1]).astype(bool)
    support[top_ft_ids] = True

    # Plot a boxplot for each feature, showing its distribution in each class
    if plot:
        plot_feature_boxplots(
            feat.iloc[:, top_ft_ids[:k_to_plot]], y_class, mrmr_scores,
            figsize=figsize, saveto=saveto,
            close_after_plotting=close_after_plotting)

    return feat.columns[top_ft_ids].to_list(), mrmr_scores, support



def top_feat_in_PCs(X, pc=0, scale=False, k='auto', feat_names=None):
    """
    Runs PCA and gives the top k contributing features for the specified
    principal component (by default the first component).

    """
    import pandas as pd

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
            tierpsytools.preprocessing.scaling_class.scalingClass is used
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
    from tierpsytools.preprocessing.scaling_class import scalingClass

    if k_to_plot is None:
        plot = False
    else:
        plot = True

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
    from sklearn.linear_model import LogisticRegression

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

