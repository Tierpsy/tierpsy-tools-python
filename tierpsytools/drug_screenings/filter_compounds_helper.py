#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:43:30 2020

@author: em812
"""
import pandas as pd
import numpy as np
import pdb
from joblib import Parallel, delayed

def _stats_test_basic(X, y, test,  **kwargs):
    # Local function for parallel processing of univariate tests for each drug
    from joblib import Parallel, delayed

    def _one_fit(ift, samples, **kwargs):
        samples = [s[~np.isnan(s)] for s in samples if not all(np.isnan(s))]
        try:
            rp = test(*samples, **kwargs)
        except Exception as err:
            print(err)
            rp = (np.nan, np.nan)
        return ift, rp

    parallel = Parallel(n_jobs=-1, verbose=True)
    func = delayed(_one_fit)

    try:
        res = parallel(
            func(ift, [sample[:,ift]
                  for sample in [np.array(X[y==iy]) for iy in np.unique(y)]],
                 **kwargs)
            for ift in range(X.shape[1]))
    except Exception as err:
        print(err)
        pdb.set_trace()

    order = [ift for ift,(r,p) in res]
    rs = np.array([r for ift,(r,p) in res])
    ps = np.array([p for ift,(r,p) in res])

    return rs[order], ps[order]


def _stats_test_multifeat(X, y, test, **kwargs):
    # Local function for tests that support many featurres at the same time

    samples = [X[y==iy] for iy in np.unique(y)]
    try:
        stats, pvals = test(*samples, axis=0, **kwargs)
    except Exception as err:
        print(err)
        stats, pvals = (np.ones(X.shape[1])*np.nan, np.ones(X.shape[1])*np.nan)

    return stats, pvals

def _stats_test(X, y, test, multifeat=False, **kwargs):

    if multifeat:
        return _stats_test_multifeat(X, y, test, **kwargs)
    else:
        return _stats_test_basic(X, y, test, **kwargs)


def _low_effect_univariate(
        X, drug_dose, comparison_type='multiclass', control_dose=.0,
        test='ANOVA', multitest_method='fdr_by', fdr=0.05, n_jobs=-1):
    """
    Test whether a single compound has siginificant effects compared to the
    control using univariate tests for each feature.
    Each feature is tested using one of the methods 'ANOVA', 'Kruska-Wallis'
    or 'Wilcoxon-rank test'.
    The pvalues from the different features are corrected for multiple
    comparisons using the multitest methods of statsmodels.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    drug_dose : TYPE
        DESCRIPTION.
    comparison_type : TYPE, optional
        DESCRIPTION. The default is 'multiclass'.
    control_dose : float, optional. The default is .0.
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
    from scipy.stats import kruskal, ranksums, f_oneway, ttest_ind
    from functools import partial

    if comparison_type=='multiclass' and np.unique(drug_dose).shape[0]>2:
        if test.startswith('Wilkoxon') or test == 't-test':
            raise ValueError(
                """
            The Wilkoxon rank sum test cannot be used to compare between
            more than two groups. Use a different test or the
            binary_each_dose comparison_method instead.
                """)

    if isinstance(X, pd.DataFrame):
        ft_names = X.columns
        X = X.values

    # Create the function that will test every feature of a given drug
    if test == 'ANOVA':
        func = partial(_stats_test, test=f_oneway, multifeat=True)
    elif test.startswith('Kruskal'):
        func = partial(_stats_test, test=kruskal, nan_policy='raise', multifeat=False)
    elif test.startswith('Wilkoxon'):
        func = partial(_stats_test, test=ranksums, multifeat=False)
    if test == 't-test':
        func = partial(_stats_test, test=ttest_ind, nan_policy='omit', multifeat=True)

    # For each dose get significant features
    if comparison_type=='multiclass':
        _, pvals = func(X, drug_dose)
        pvals = pd.Series(pvals, name='all')

    elif comparison_type=='binary_each_dose':
        if not np.isin(control_dose, np.array(drug_dose)):
            raise ValueError('control_dose not found in the drug_dose array.')
        doses = np.unique(drug_dose[drug_dose!=control_dose])

        pvals=[]
        for idose, dose in enumerate(doses):

            mask = np.isin(drug_dose,[control_dose, dose])
            _, _pvals = func(X[mask], drug_dose[mask])

            pvals.append(_pvals)

        # cast into recarray with doses as names
        pvals = pd.DataFrame(np.array(pvals).T, index=ft_names, columns=doses)

    else:
        raise ValueError('Comparison type not recognised.')

    return pvals


def _low_effect_LMM(
        X, drug_dose, random_effect, control_dose=.0,
        fdr=0.05, multitest_method='fdr_by', comparison_type='multiclass',
        n_jobs=-1
        ):
    """
    Test whether a single compound has siginificant effects compared to the
    control, taking into account one random effect.
    Each feature of is tested independently using a Linear Mixed Model with
    fixed slope and variable intercept to account for the random effect.
    The pvalues from the different features are corrected for multiple
    comparisons using the multitest methods of statsmodels.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    drug_dose : TYPE
        DESCRIPTION.
    random_effect : TYPE
        DESCRIPTION.
    control_dose : float, optional. The default is .0.
        The dose of the control points in drug_dose.
    fdr : TYPE, optional
        DESCRIPTION. The default is 0.05.
    multitest_method : TYPE, optional
        DESCRIPTION. The default is 'fdr_by'.
    n_jobs : TYPE, optional
        DESCRIPTION. The default is -1.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    from tqdm import tqdm
    import statsmodels.formula.api as smf

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    feat_names = X.columns.to_list()
    X = X.assign(drug_dose=drug_dose).assign(random_effect=random_effect)

    # select only the control points that belong to groups that have
    # non-control members
    groups = np.unique(random_effect[drug_dose!=control_dose])
    X = X[np.isin(random_effect, groups)]

    # Convert the independent variable to categorical if you want to compare by dose
    if comparison_type=='binary_each_dose':
        X['drug_dose'] = X['drug_dose'].astype(str)

    # Intitialize pvals as series or dataframe
    # (based on the number of comparisons per feature)
    if comparison_type == 'multiclass':
        pvals = pd.Series(index=feat_names, dtype=float)
    elif comparison_type == 'binary_each_dose':
        doses = np.unique(drug_dose[drug_dose!=control_dose])
        pvals = pd.DataFrame(index=feat_names, columns=doses)
    else:
        raise ValueError('Comparison type not recognised.')

    # Local function that tests a single feature
    def _one_fit(ft, data):
        # remove dose groups with only one member
        try:
            data = pd.concat(
                [x for _,x in data.groupby(by=['drug_dose', 'random_effect']) if x.shape[0]>1]
                )
        except:
            return ft,np.nan

        # Define LMM
        md = smf.mixedlm("{} ~ drug_dose".format(ft), data,
                         groups=data['random_effect'].astype(str),
                         re_formula="")
        # Fit LMM
        try:
            mdf = md.fit()
            pval = mdf.pvalues[[k for k in mdf.pvalues.keys() if k.startswith('drug_dose')]]
            pval = pval.min()
        except:
            pval = np.nan

        return ft, pval

    ## Fit LMMs for each feature
    # -Run single job alternative
    # (using a for loop is faster than launching a single job with joblib)
    if n_jobs==1:
        for ft in tqdm (feat_names, desc="Testing features…", ascii=False):
            _, pvals.loc[ft] = _one_fit(
                ft, X[[ft, 'drug_dose', 'random_effect']].dropna(axis=0))
    # -Run parallel jobs alternative
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=True)
        func = delayed(_one_fit)

        res = parallel(
            func(ft, X[[ft, 'drug_dose', 'random_effect']].dropna(axis=0))
            for ft in feat_names)
        for ft, _pval in res:
            pvals.loc[ft] = _pval

    return pvals

