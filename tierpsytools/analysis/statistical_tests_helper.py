#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:24:40 2021

@author: em812
"""
import numpy as np
from functools import partial
import pdb

#%% Define statistical test functions
def get_test_fun(test, vectorized=False):
    from scipy.stats import kruskal, mannwhitneyu, f_oneway, ttest_ind

    if test == 'ANOVA':
        func = partial(stats_test, test=f_oneway, vectorized=True)
    elif test.startswith('Kruskal'):
        func = partial(stats_test, test=kruskal, nan_policy='raise', vectorized=False)
    elif test.startswith('Mann-Whitney'):
        func = partial(stats_test, test=mannwhitneyu, vectorized=False)
    elif test == 't-test':
        func = partial(stats_test, test=ttest_ind, vectorized=True)

    return func

def stats_test(X, y, test, vectorized, n_jobs=-1, **kwargs):
    """
    Function that rearranges the X,y input to the format expected by the scipy.stats
    functions and returns the results for the stastical tests for every feature of X
    in the format expected by the univariate_tests function (two arrays).
    A different type of rearrangement is needed when the scipy.stats
    function is vecotrized (can take multi-dimensional samples) and when the
    function is not vectorized. In the latter case, parallel processing is used
    to run each feature.

    Parameters
    ----------
    X : array or dataframe, shape (n_datapoints, n_features)
        The features matrix for all the data.
    y : array-like
        An array with the group labels for each datapoint.
    test : function object
        The scipy.stats function to use for the statistical tests (or any other
        function with same type of input at the scipt.stats functions).
    vectorized : boolean
        Whether the stats function in 'test' is vectorized or not.
    n_jobs : int, optional
        Number of cores to use for the parallel processing of each feature.
        If -1, then all available cores will be used.
        If vectorized=True, this parameter will be ignored.
        Deafault is -1.
    **kwargs : keyword argunements
        Any additional input arguments required by the test function.

    Returns
    -------
    stats: array-like, shape (n_features,)
        The statistic from the statistical tests for each features
    pvals: array-like, shape (n_features,)
        The p-values from the statistical tests for each features

    """
    if vectorized:
        return stats_test_vectorized(X, y, test, **kwargs)
    else:
        return stats_test_parallel(X, y, test, n_jobs, **kwargs)


def stats_test_vectorized(X, y, test, **kwargs):
    samples = [np.array(X[y==iy]) for iy in np.unique(y)]
    samples = [s for s in samples if not np.all(np.isnan(s))]
    if len(samples)<2: # if only one group after dropping nans
        return np.ones(X.shape[1])*np.nan, np.ones(X.shape[1])*np.nan

    res = test(*samples, **kwargs)
    return res[0], res[1]

def stats_test_parallel(X, y, test, n_jobs, **kwargs):
    from joblib import Parallel, delayed

    def _one_fit(ift, samples, **kwargs):
        samples = [s[~np.isnan(s)] for s in samples if not all(np.isnan(s))]
        if len(samples)<2: # if only one group after dropping nans
            return ift, (np.nan, np.nan)
        try:
            sp = test(*samples, **kwargs)
        except:
            sp = (np.nan, np.nan)
        return ift, sp

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


#%%