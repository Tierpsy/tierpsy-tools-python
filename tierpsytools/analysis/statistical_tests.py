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
            'binary_each_group': compare independently each group with the
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
    from tierpsytools.analysis.statistical_tests_helper import get_test_fun

    if not np.isin(control, np.array(y)):
        raise ValueError('control not found in the y array.')

    if np.unique(y).shape[0]<2:
        raise Exception('Only one group found in y. Nothing to compare with.')

    if test.startswith('Mann') or test == 't-test':
        if comparison_type=='multiclass' and np.unique(y).shape[0]>2:
            raise ValueError(
                """
            The {} cannot be used to compare between
            more than two groups. Use a different test or the
            binary_each_dose comparison_method instead.
                """. format(test))
        else:
            comparison_type = 'binary_each_group'

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Create the function that will test every feature of a given drug
    func = get_test_fun(test)

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

    reject, pvals = _multitest_correct(pvals, multitest_correction, alpha)

    return stats, pvals, reject

def get_effect_sizes(
        X, y, control='N2', effect_type=None,
        linked_test='ANOVA', comparison_type='multiclass'
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
    effect_type : str or None, optional
        The type of effect size to calculate.
        Available options are: 'eta_squared', 'cohen_d', 'cliffs_delta'.
        If None, the type of effect size will be decided using the linked_test
        parameter.
        Default is None.
    linked_test : str, optional
        This parameter, together with the comparison_type, will define the type
        of effect size that will be calculated, if the effect_type parameter is None.
        With this parameter the user can define the statistical test for which
        they want an effect size.
        For example, if the linked_test is 'ANOVA', this means that we want to
        compare two or more groups and see if any of them is different than the others.
        For this type of comparison, we would estimate the eta_squared effect size.
        Available options are:
        'Mann-Whitney', 'Kruskal-Wallis', 'ANOVA', 't-test'.
        When the effect_type parameter is defined (when it is not None), then this
        parameter is ignored.
        The default is 'ANOVA'.
    comparison_type : str, optional
        Available options are: 'multiclass' or 'binary_each_group'.
        This parameter is required only when the effect_type is eta_squared
        or when the effect_type is None and the linked_test is 'ANOVA'. In any other
        case, this parameter is ignored.
        When 'multiclass', the null hypothesis is that all groups come from the same
        distribution, the alternative hypothesis is that at least one of them
        comes from a different distribution. One effect type will be estimated,
        regardless the number of groups in y.
        When 'binary_each_group', then each group will be compared with the control
        group defined in the 'control' parameter. One separate effect size will
        be returned for each group.
        The default is 'multiclass'.

    The following table is a guide to the correspondance between effect size
    type and statistical tests :
            linked_test    |  comparison_type     |   effect_type
            -------------------------------------------------------
            ANOVA          |  binary_each_group   |   eta_squared
            ANOVA          |  multiclass          |   eta_squared
            t-test         |  N/A                 |   cohen_d
            Kruskal-Wallis |  N/A                 |   cliffs_delta
            'Mann-Whitney' |  N/A                 |   cliffs_delta

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    effect : dataframe
        The effect sizes estimated for each feature.

    """
    from tierpsytools.analysis.statistical_tests_helper import \
        eta_squared_ANOVA, cohen_d, cliffs_delta

    if not np.isin(control, np.array(y)):
        raise ValueError('control not found in the comparison_variable array.')

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    groups = np.unique(y[y!=control])

    # Get the effect_type based on linked_test
    if effect_type is None:
        if not isinstance(linked_test,str):
            raise Exception('Input type is not recognised for the linked_test '+
                            'parameter. See available options in docstring.')
        elif linked_test=='ANOVA':
            effect_type = 'eta_squared'
        elif linked_test == 't-test':
            effect_type = 'cohen_d'
        elif linked_test.startswith('Kruskal') or linked_test.startswith('Mann'):
            effect_type = 'cliffs_delta'
        else:
            raise Exception('Input not recognised for the linked_test parameter. '+
                            'See available options in the function docstring.')
    elif not np.isin(effect_type, ['eta_squared', 'cohen_d', 'cliffs_delta']):
        raise Exception('Input not recognised for the effect_type parameter. '+
                        'See available options in the function docstring.')

    # Get the effect sizes for each type of test
    if effect_type == 'eta_squared':
        if comparison_type=='multiclass':
            effect = pd.Series(index=X.columns)
            samples = [x for ix,x in X.groupby(by=y)]
            for ft in X.columns:
                effect[ft] = eta_squared_ANOVA(*[s.loc[~s[ft].isna(), ft] for s in samples])
            effect = pd.DataFrame(effect, columns=[effect_type])
            return effect
        elif comparison_type=='binary_each_group':
            effect = pd.DataFrame(index=X.columns, columns=groups)
            for igrp, grp in enumerate(groups):
                mask = np.isin(y,[control, grp])
                samples = [x for ix,x in X[mask].groupby(by=y[mask])]
                for ft in X.columns:
                    effect.loc[ft, grp] = eta_squared_ANOVA(
                        *[s.loc[~s[ft].isna(), ft] for s in samples])

    elif effect_type =='cohen_d':
        effect = {}
        for igrp, grp in enumerate(groups):
            mask = np.isin(y,[control, grp])
            effect[grp] = cohen_d(*[x for ix,x in X[mask].groupby(by=y[mask])])
        effect = pd.DataFrame(effect, index=X.columns)
        return effect

    elif effect_type == 'cliffs_delta':
        effect = pd.DataFrame(index=X.columns, columns=groups)
        for igrp, grp in enumerate(groups):
            mask = np.isin(y,[control, grp])
            samples = [x for ix,x in X[mask].groupby(by=y[mask])]
            for ft in X.columns:
                effect.loc[ft, grp] = cliffs_delta(
                    *[s.loc[~s[ft].isna(), ft] for s in samples])

    return effect


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
    c_reject = np.zeros(mask.shape, dtype=bool)
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
    """
    Get the confidence interval (CI) of a metric using bootstrapping.

    Parameters
    ----------
    x : array-like
        a sample.
    func : callable (function object)
        the function that estimated the metric (for example np.mean, np.median, ...).
    n_boot : int
        number of sub-samples to use for the bootstrap estimate.
    which_ci : float, optional
        A number between 0 and 100 that defines the confidence interval.
        The default is 95, which means that there is 95% probability the metric
        will be inside the limits of the confidence interval.
    axis : int or None, optional
        Will pass axis to func as a keyword argument.
        The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    from seaborn.algorithms import bootstrap
    from seaborn.utils import ci

    boot_distribution = bootstrap(x, func=func, n_boot=n_boot, axis=axis)

    return ci(boot_distribution, which=which_ci, axis=axis)
