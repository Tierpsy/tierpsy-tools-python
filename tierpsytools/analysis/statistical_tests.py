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
        multitest_correction='fdr_by', alpha=0.05,
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
    X : dataframe or array
        feature matrix.
    y : array-like
        labes of the samples for the comparison.
    control : float, optional. The default is .0.
        The y entry for the control points.
        Must provide control y if the comparison_type is 'binary_each_group'.
    test : str, optional
        Type of statistical test to perform. The options are:
        'ANOVA', 'Kruskal-Wallis', 'Mann-Whitney test' or 't-test'.
        The default is 'ANOVA'.
    comparison_type : string, optional
        The type of comparison to make. This parameter is ignored if there are
        only two unique groups in y.
        The options are:
            'multiclass': perform comparison among all the unique groups in y.
            Compatible with tests that can handle more than two groups ('ANOVA', 'Kruskal)
            'binary_each_class': compare independently each group with the
            control group and return all p-values from the different comparisons.
            Compatible with all tests.
        The default is 'multiclass'.
    multitest_correction : string or None, optional
        Method to use to correct p-values for multiple comparisons. The options
        are the ones available in statsmodels.stats.multitest.multipletests.
        The default is 'fdr_by'.
    alpha : float, optional
        The significant level for the corrected p-values. The default is 0.05.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    stats:

    pvals:
        The p-values for all the comparisons made.
    reject: array of bool
        A mask array defining the significant comparisons.

    """
    from scipy.stats import kruskal, mannwhitneyu, f_oneway, ttest_ind
    from functools import partial

    if not np.isin(control, np.array(y)):
        raise ValueError('control not found in the y array.')

    if test.startswith('Mann') or test == 't-test':
        if comparison_type=='multiclass' and np.unique(y).shape[0]>2:
            raise ValueError(
                """
            The {} cannot be used to compare between
            more than two groups. Use a different test or the
            binary_each_dose comparison_method instead.
                """. format(test))
        else:
            comparison_type = 'binary_each_class'

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Local function for parallel processing of univariate tests for each drug
    def stats_test(X, y, test, **kwargs):
        from joblib import Parallel, delayed

        def _one_fit(ift, samples, **kwargs):
            samples = [s[~np.isnan(s)] for s in samples if not all(np.isnan(s))]
            # TODO
            if len(samples)<2: # if only one group --> put it outside
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

    elif comparison_type=='binary_each_class':
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

    reject, pvals = _multitest_correct(pvals, multitest_correction, alpha)

    return stats, pvals, reject

def get_effect_sizes(
        X, y, control='N2',
        test='ANOVA', comparison_type='multiclass'
        ):
    """
    Get the effect sizes of statistical comparisons between groups defined in y.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The features matrix.
    y : list-like, shape=(n_samples)
        The dependent variable - the variable that defines the groups to compare.
    control : str
        The group in y which is considered the control group.
    comparison_type : str, optional
        'multiclass' or 'binary_each_group'. The default is 'multiclass'.
        When 'multiclass', the null hypothesis is that all groups come from the same
        distribution, the alternative hypothesis is that at least one of them
        comes from a different distribution.
        When 'binary_each_group', then each group will be compared with the control
        group defined in the 'control' parameter.
    test : str, optional
        Mann-Whitney', 'Kruskal-Wallis', 'ANOVA' or 't-test'. The default is 'ANOVA'.

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

    elif test=='t-test':
        effect = {}
        for igrp, grp in enumerate(groups):
            mask = np.isin(y,[control, grp])
            effect[grp] = cohen_d(*[x for ix,x in X[mask].groupby(by=y)])
        effect = pd.DataFrame(effect, index=X.columns)

    else:
        if test=='Mann-Whitney' or test=='Kruskal-Wallis':
            func = cliffs_delta
        elif test=='ANOVA':
            func = eta_squared_ANOVA

        effect = pd.DataFrame(index=X.columns, columns=groups)
        for igrp, grp in enumerate(groups):
            mask = np.isin(y,[control, grp])
            samples = [x for ix,x in X[mask].groupby(by=y)]
            for ft in X.columns:
                effect.loc[ft, grp] = func(*[s.loc[~s[ft].isna(), ft] for s in samples])

    return effect

#%% Effect size functions
def cohen_d(x,y, axis=0):
    """ Return the cohen d effect size for t-test

    """
    from numpy import nanstd, nanmean, sqrt

    x = np.array(x)
    y = np.array(y)

    if len(x.shape)==1:
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (nanmean(y) - nanmean(x)) / sqrt(((nx-1)*nanstd(x, ddof=1) ** 2 + (ny-1)*nanstd(y, ddof=1) ** 2) / dof)
    elif len(x.shape)==2:
        if axis not in [0,1]:
            raise Exception('Axis out of range.')
        nx = x.shape[axis]
        ny = y.shape[axis]
        dof = nx + ny - 2
        return (nanmean(y,axis=axis) - nanmean(x,axis=axis)) / \
            sqrt(((nx-1)*nanstd(x, axis=axis, ddof=1) ** 2 + (ny-1)*nanstd(y, axis=axis, ddof=1) ** 2) / dof)
    else:
        raise Exception('Samples must be 1D or 2D.')


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

#%% Bootstrapping
def bootstrapped_ci(x, func, n_boot, which_ci=95, axis=None):
    from seaborn.algorithms import bootstrap
    from seaborn.utils import ci

    boot_distribution = bootstrap(x, func=func, n_boot=n_boot, axis=axis)

    return ci(boot_distribution, which=which_ci, axis=axis)
