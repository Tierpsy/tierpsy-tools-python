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

def univariate_tests(
        X, y, control='N2', test='ANOVA',
        comparison_type='multiclass',
        multitest_correction='fdr_by', fdr=0.05,
        n_jobs=-1):
    """
    Test whether a single compound has siginificant effects compared to the
    control using univariate tests for each feature.
    Each feature is tested using one of the methods 'ANOVA', 'Kruskal-Wallis',
    'Mann-Whitney test' or 't-test'.
    The pvalues from the different features are corrected for multiple
    comparisons using the multitest methods of statsmodels.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    comparison_type : TYPE, optional
        DESCRIPTION. The default is 'multiclass'.
    control : float, optional. The default is .0.
        The drug_dose entry for the control points.
        Must provide control dose if the comparison_type is 'binary_each_group'.
    test : TYPE, optional
        DESCRIPTION. The default is 'ANOVA'.
    multitest_correction : string or None, optional
        DESCRIPTION. The default is 'fdr_by'.
    fdr : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    from scipy.stats import kruskal, mannwhitneyu, f_oneway, ttest_ind
    from functools import partial

    if not np.isin(control, np.array(y)):
        raise ValueError('control not found in the y array.')

    if test.startswith('Wilkoxon') or test == 't-test':
        if comparison_type=='multiclass' and np.unique(y).shape[0]>2:
            raise ValueError(
                """
            The Wilkoxon rank sum test cannot be used to compare between
            more than two groups. Use a different test or the
            binary_each_dose comparison_method instead.
                """)
        else:
            comparison_type = 'binary_each_group'

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Local function for parallel processing of univariate tests for each drug
    def stats_test(X, y, test, **kwargs):
        from joblib import Parallel, delayed

        def _one_fit(ift, samples, **kwargs):
            samples = [s[~np.isnan(s)] for s in samples if not all(np.isnan(s))]
            if len(samples)<2:
                return ift, (np.nan, np.nan)
            return ift, test(*samples, **kwargs)

        parallel = Parallel(n_jobs=n_jobs, verbose=True)
        func = delayed(_one_fit)

        try:
            res = parallel(
                func(ift, [sample[:,ift]
                      for sample in [np.array(X[y==iy]) for iy in np.unique(y)]],
                     **kwargs)
                for ift in range(X.shape[1]))
        except:
            pdb.set_trace()

        order = [ift for ift,(r,p) in res]
        rs = np.array([r for ift,(r,p) in res])
        ps = np.array([p for ift,(r,p) in res])

        return rs[order], ps[order]

    # Create the function that will test every feature of a given drug
    if test == 'ANOVA':
        func = partial(stats_test, test=f_oneway)
    elif test.startswith('Kruskal'):
        func = partial(stats_test, test=kruskal, nan_policy='raise')
    elif test.startswith('Mann-Whitney'):
        func = partial(stats_test, test=mannwhitneyu)
    if test == 't-test':
        func = partial(stats_test, test=ttest_ind)

    # For each dose get significant features
    if comparison_type=='multiclass':
        stats, pvals = func(X, y)
        pvals = pd.DataFrame(pvals.T, index=X.columns, columns=[test])
        stats = pd.DataFrame(stats.T, index=X.columns, columns=[test])

    elif comparison_type=='binary_each_group':
        groups = np.unique(y[y!=control])

        pvals=[]
        stats=[]
        for igrp, grp in enumerate(groups):

            mask = np.isin(y,[control, grp])
            _stats, _pvals = func(X[mask], y[mask])

            pvals.append(_pvals)
            stats.append(_stats)
        pvals = pd.DataFrame(np.array(pvals).T, index=X.columns, columns=groups)
        stats = pd.DataFrame(np.array(stats).T, index=X.columns, columns=groups)
    else:
        raise ValueError('Comparison type not recognised.')

    reject, pvals = _multitest_correct(pvals, multitest_correction, fdr)

    return stats, pvals, reject

def get_effect_sizes(
        X, y, control='N2',
        test='ANOVA', comparison_type='multiclass',
        n_jobs=-1):
    """
    Test whether a single compound has siginificant effects compared to the
    control using univariate tests for each feature.
    Each feature is tested using one of the methods 'ANOVA', 'Kruskal-Wallis',
    'Mann-Whitney' or 't-test'.
    The pvalues from the different features are corrected for multiple
    comparisons using the multitest methods of statsmodels.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    drcomparison_variableug_dose : TYPE
        DESCRIPTION.
    comparison_type : TYPE, optional
        DESCRIPTION. The default is 'multiclass'.
    control : float, optional. The default is .0.
        The drug_dose entry for the control points.
        Must provide control dose if the comparison_type is 'binary_each_dose'.
    test : TYPE, optional
        DESCRIPTION. The default is 'ANOVA'.
    multitest_method : TYPE, optional
        DESCRIPTION. The default is 'fdr_by'.
    fdr : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    if not np.isin(control, np.array(y)):
        raise ValueError('control not found in the comparison_variable array.')

    if comparison_type=='multiclass' and np.unique(y).shape[0]>2:
        if test.startswith('Mann') or test == 't-test':
            raise ValueError(
                """
            The Mann-Whitney test cannot be used to compare between
            more than two groups. Use a different test or the
            binary_each_dose comparison_method instead.
                """)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    groups = np.unique(y[y!=control])

    # For each dose get significant features
    if test=='ANOVA' and comparison_type=='multiclass':
        effect = pd.Series(index=X.columns)
        samples = [x for ix,x in X.groupby(by=y)]
        for ft in X.columns:
            effect[ft] = eta_squared_ANOVA(*[s.loc[~s[ft].isna(), ft] for s in samples])
        effect = pd.DataFrame(effect, columns=['_'.join([test,'effect_size'])])
    else:
        if test=='Mann-Whitney' or test=='Kruskal-Wallis':
            func = cliffs_delta
        elif test=='ANOVA':
            func = eta_squared_ANOVA
        elif test=='t-test':
            func = cohen_d

        effect = pd.DataFrame(index=X.columns, columns=groups)
        for igrp, grp in enumerate(groups):
            mask = np.isin(y,[control, grp])
            samples = [x for ix,x in X[mask].groupby(by=y)]
            for ft in X.columns:
                effect.loc[ft, grp] = func(*[s.loc[~s[ft].isna(), ft] for s in samples])

    return effect

#%% Effect size functions
def cohen_d(x,y):
    """ Return the cohen d effect size for t-test

    """
    from numpy import nanstd, nanmean, sqrt

    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (nanmean(x) - nanmean(y)) / sqrt(((nx-1)*nanstd(x, ddof=1) ** 2 + (ny-1)*nanstd(y, ddof=1) ** 2) / dof)


def eta_squared_ANOVA( *args):
    """ Return the eta squared as the effect size for ANOVA

    """
    return( float( __ss_between_( *args) / __ss_total_( *args)))

def cliffs_delta(lst1, lst2):

    """Returns delta and true if there are more than 'dull' differences"""
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in _runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
    return d

def __concentrate_( *args):
    """ Concentrate input list-like arrays

    """
    v = list( map( np.asarray, args))
    vec = np.hstack( np.concatenate( v))
    return( vec)

def __ss_total_( *args):
    """ Return total of sum of square

    """
    vec = __concentrate_( *args)
    ss_total = sum( (vec - np.mean( vec)) **2)
    return( ss_total)

def __ss_between_( *args):
    """ Return between-subject sum of squares

    """
    # grand mean
    grand_mean = np.mean( __concentrate_( *args))

    ss_btwn = 0
    for a in args:
        ss_btwn += ( len(a) * ( np.mean( a) - grand_mean) **2)

    return( ss_btwn)

def _runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two

#%% Correct for multiple comparisons

def _multitest_correct(pvals, multitest_method, fdr):
    """
    Multiple comparisons correction of pvalues from univariate tests.
    Ignores nan values.
    Deals with two options:
        - 1D array of pvalues (one comparison per feature)
        - 2D array of pvalues (multiple comparisons per feature)

    Parameters
    ----------
    pvals : pandas series shape=(n_features,) or
            pandas dataframe shape=(n_features, n_doses)
        The pandas structure containing the pvalues from all the statistical
        tests done for a single drug.
    multitest_method : string
        The method to use in statsmodels.statis.multitest.multipletests function.
    fdr : float
        False discovery rate.

    Returns
    -------
    c_reject : pandas series shape=(n_features)
        Flags indicating rejected null hypothesis after the correction for
        multiple comparisons. The null hypothesis for each feature is that the
        feature is not affected by the compound.
    c_pvals : pandas series shape=(n_features)
        The corrected pvalues for each feature. When each dose was tested
        separately, the min pvalue is stored in this output.

    """
    from statsmodels.stats.multitest import multipletests

    if multitest_method is None:
        return pvals<fdr, pvals

    if np.all(pd.isnull(pvals.values)):
        return pvals, pvals

    # Mask nans in pvalues
    if len(pvals.shape) == 1:
        mask = ~pd.isnull(pvals.values)
    else:
        mask = ~pd.isnull(pvals.values.reshape(-1))

    # Initialize array with corrected pvalues and reject hypothesis flags
    c_reject = np.ones(mask.shape)*np.nan
    c_pvals = np.ones(mask.shape)*np.nan

    # Make the correction with the chosen multitest_method
    try:
        c_reject[mask], c_pvals[mask],_,_ = multipletests(
            pvals.values.reshape(-1)[mask], alpha=fdr, method=multitest_method,
            is_sorted=False, returnsorted=False
            )
    except:
        pdb.set_trace()

    if len(pvals.shape) == 2:
        # When multiple comparisons per feature, reshape the corrected arrays
        c_reject = c_reject.reshape(pvals.shape)
        c_pvals = c_pvals.reshape(pvals.shape)
        # Convert the corrected pvals and the flags array to pandas series and
        # add feature names as index
        c_pvals = pd.DataFrame(c_pvals, index=pvals.index, columns=pvals.columns)
        c_reject = pd.DataFrame(c_reject, index=pvals.index, columns=pvals.columns)
    else:
        # Convert the corrected pvals and the flags array to pandas series and
        # add feature names as index
        c_pvals = pd.Series(c_pvals, index=pvals.index)
        c_reject = pd.Series(c_reject, index=pvals.index)

    return c_reject, c_pvals