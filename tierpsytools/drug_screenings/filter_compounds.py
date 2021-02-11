#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:28:09 2020

@author: em812
"""
import numpy as np
import pandas as pd
import pdb
import warnings
from tierpsytools.analysis.statistical_tests import _multitest_correct

def remove_MOAs_based_on_drug_count(
        feat, meta,
        min_n_compounds=3, return_compounds=False,
        moa_column='MOA_group', ignore=[-1,0],
        drugname_column='drug_type'
        ):
    """
    Removes the MOA groups that have n_compounds smaller than the min value
    defined.
    param:
        feat : dataframe shape = (n_samples x n_features)
            feature dataframe (only features, not drug_names or drug_doses)
        meta : dataframe shape = (n_samples x n_metadata_cols)
            dataframe containing drug_name, drug_dose, file_id and MOA info
            for each sample in feat
        min_n_compounds : int, optional
            Min number of compounds in a MOA group, to keep it
        return_compounds : bool, optional
            If True, the feat and meta of the removed compounds will be returned
            together with the filtered feat and meta
        moa_column: string, optional
            Specifies the metadata column to use to count the compounds per moa.
        ignore : list
            List of values in moa_column to ignore in the filtering
            (usually the values that correspond to DMSO, NoCompound, which we
            want to keep in our data)
        drugname_column: string, optional
            Specifies the metadata column that contains the individual compound
            names.
    return:
        mfeat: dataframe
            The filtered features dataframe
        mmeta: dataframe
            The filtered metadata dataframe matching mfeat
        sfeat: dataframe, optional
            The removed part of feat
        smeta: dataframe, optional
            The removed part of meta
    """

    # - remove moas with a fewer compounds than min_n_coumpounds
    moas_to_keep = meta.groupby(by=moa_column)[drugname_column].nunique()>=min_n_compounds
    moas_to_keep = moas_to_keep[moas_to_keep.values].index.to_list()

    if return_compounds:
        sfeat = feat.loc[~meta[moa_column].isin(moas_to_keep), :]
        smeta = meta.loc[~meta[moa_column].isin(moas_to_keep), :]

    feat = feat.loc[meta[moa_column].isin(moas_to_keep), :]
    meta = meta.loc[meta[moa_column].isin(moas_to_keep), :]

    if return_compounds:
        return feat, meta, sfeat, smeta
    else:
        return feat, meta

def compare_drug_to_control_univariate(
        X, drug_dose, random_effect=None, control_dose=.0,
        test='ANOVA', comparison_type='multiclass',
        fdr=0.05, multitest_method='fdr_by', n_jobs=-1,
        ):
    from tierpsytools.drug_screenings.filter_compounds_helper import \
        _low_effect_univariate, _low_effect_LMM

    # Check input
    if any([test.startswith(t) for t in ['ANOVA', 't-test', 'Kruskal','Wilcoxon']]):
        if comparison_type is None:
            raise ValueError('Must define the coparison_type for the '+
                             'univariate {} tests.'.format(test))
        pvals = _low_effect_univariate(
            X, drug_dose, comparison_type=comparison_type,
            control_dose=control_dose, test=test,
            multitest_method=multitest_method, fdr=fdr, n_jobs=n_jobs)

    elif test=='LMM':
        if random_effect is None:
            raise ValueError('Must give the random_effect variable for the '+
                             'LMM tests.')

        pvals = _low_effect_LMM(
            X, drug_dose, random_effect, control_dose=control_dose,
            fdr=fdr, multitest_method=multitest_method,
            comparison_type=comparison_type, n_jobs=n_jobs)

    else:
        raise ValueError('Test type not recognised.')

    reject, pvals = _multitest_correct(pvals, multitest_method, fdr)

    # When doses were tested separately, keep only the min pvalue and a
    # unique reject flag for each feature
    if len(pvals.shape) == 2:
        reject = pd.Series(np.any(reject.reshape(pvals.shape), axis=1), index=pvals.index)
        pvals = pd.Series(np.min(pvals.reshape(pvals.shape), axis=1), index=pvals.index)

    return reject, pvals


def compounds_with_low_effect_univariate(
        feat, drug_name, drug_dose=None, random_effect=None,
        control='DMSO',
        test='ANOVA', comparison_type='multiclass',
        multitest_method='fdr_by', fdr=0.05,
        ignore_names=['NoCompound'],
        return_pvals=False, n_jobs=-1
        ):
    """
    Detects drugs with low effect and drugs with significant effect compared to
    the control. A drug is considered to have a significant effect when at least
    one feature is significantly different between the drug at any dose and the
    control.
    The statistical significance of the difference between compound and
    control is assessed based on individual statistical tests for each feature
    (options for parametric tests, non-parameteric tests and tests based on
     linear mixed models accounting for one random effect).
    The Benjamini-Yuketieli method is used to control the false discovery rate
    within each drug.

    Parameters
    ----------
    feat : dataframe
        feature dataframe, shape=(n_samples, n_features)
    drug_name : array-like, shape=(n_samples)
        defines the type of drug in each sample
    drug_dose : array-like or None, shape=(n_samples)
        defines the drug dose in each sample.
        If None, it is assumed that each drug was tested at a single dose.
    random_effect : array-like, shape=(n_samples), optional. Default is None.
        The variable that is considered to be a random effect, when the LMM
        test is used. Must be a categorical variable with categories that
        have members both in the control group and the drug groups.
        When another test type is chosen (test!='LMM'), this parameter is
        ignored.
    control : str
        the name of the control samples in the drug_name array
    match_control_by : array-like, shape=(n_samples), optional. Default is None.
        If this variable is defined, then the samples are compared only with
        the control points that have the same value for this variable.
        For example, if match_control_by is the day of the experiment, we
        compare the drug samples with the control points from the same day of
        experiment.
    test : str, options: ['ANOVA', 't-test', 'Kruskal_Wallis', 'Wilkoxon_Rank_Sum', 'LMM']
        The type of statistical test to perform for each feature.
    comparison_type : str, options: ['multiclass', 'binary_each_dose']
        defines the groups seen in the statistical test.
        - If 'multiclass', then the controls and each drug dose are all
        considered separate groups.
        - If 'binary_each_dose', then separate tests are performed for each
        dose in every feature. Each test compares one dose to the controls.
        If any of the tests reaches the significance thresshold (after
        correction for multiple comparisons), then the feature is
        considered significant.
        * When only one dose was tested per compound, then this parameter can take any value.
    multitest_method : string or None
        The method to use in statsmodels.statis.multitest.multipletests function.
        If None, no correction is done.
    fdr : float < 1.0 and > 0.0
        false discovery rate
    ignore_names : list or None, optional
        list of names from the drug_name array to ignore in the comparisons
        (in addition to the control)
    return_pvals: bool
        Wether or not to return the pvalues of all the comparisons
    n_jobs: int
        Number of jobs for parallel processing.

    Return
    ------
    signif_effect_drugs : list
    low_effect_drugs : list
    significant : dictionary
        mask of significant features for each drug name
    """
    if drug_dose is None:
        drug_dose = np.ones(drug_name.shape)
        drug_dose[drug_name==control] = 0

    # get the dose entry for the control
    control_dose = np.unique(drug_dose[drug_name==control])
    if control_dose.shape[0]>1:
        raise ValueError('The dose assinged to the control data is not '+
                         'unique.')
    control_dose = control_dose[0]

    # Ignore the control and any names defined by user
    if ignore_names is None:
        ignore_names = []
    ignore_names.append(control)

    # Get the list of drug names to test
    drug_names = np.array([drug for drug in np.unique(drug_name)
                           if drug not in ignore_names])

    # Initialize list to store pvals and reject flags for each drug
    pvals = []
    reject = []

    # Loop over drugs to test
    for idrug,drug in enumerate(drug_names):
        print('Univariate tests for compound {}...'.format(drug))

        # get mask for the samples of the drug and the control
        mask = np.isin(drug_name, [drug, control])

        # mask the random effect variable if it exists
        randeff = random_effect[mask] if random_effect is not None else None

        # Run all the tests
        c_reject, c_pvals = compare_drug_to_control_univariate(
            feat[mask], drug_dose[mask], random_effect=randeff,
            control_dose=control_dose,
            test=test, comparison_type=comparison_type,
            multitest_method=multitest_method, fdr=fdr, n_jobs=n_jobs)

        # Store the corrected pvalues and rejected-null-hypothesis flags
        pvals.append(c_pvals.rename(drug))
        reject.append(c_reject.rename(drug))

    # Create dataframe with all pvalues and rejected null hypothesis flags
    reject = pd.DataFrame(reject)
    pvals = pd.DataFrame(pvals)
    drug_names = pvals.index.to_numpy()

    # Check for drugs that gave only nan pvals and flag them as error drugs
    error = pvals.isna().all(axis=1).values
    error_drugs = drug_names[error]

    # Check for drugs that have at least one feature where the null hypothesis
    # was rejected and flag them as significant effect drugs.
    # Flag the remaining drugs as low effect drugs.
    signif_effect = reject.any(axis=1).values

    signif_effect_drugs = drug_names[signif_effect & ~error]
    low_effect_drugs = drug_names[~error & ~signif_effect]

    if return_pvals:
        return signif_effect_drugs, low_effect_drugs, error_drugs, reject, pvals
    else:
        return signif_effect_drugs, low_effect_drugs, error_drugs

def compounds_with_low_effect_classification(
        feat, drug_name, drug_dose, control='DMSO', metric='recall',
        pval_thres=0.05, estimator = None, n_folds=5, ignore_names = None
        ):
    """
    Find drugs with significant effect and drugs with low effect based on the
    ability of a classifier to distinguish the drug doses from the control
    with a performance better than chance.
    param:
        feat : dataframe
            feature dataframe
        drug_name : array-like
            the drug name per row in feat
        drug_dose : array-like
            the drug dose per row in feat
        control : str
            the name of the control samples in the drug_name array
        metric : str, options=['recall', 'mcc', 'balanced_accuracy']
            the classification metric to use for the comparison.
        pval_thres : float
            the significance threshold for the comparison of the cv score obtained
            with the classifie to the distribution of random scores based on
            permutations. If th pvalue from this comparison is smaller than
            the pval_thres, then the drug dose is considered to have a real effect.
        cutoff : float or None
            The score value that will be the threshold to consider a dose
            significantly different to the control (if score>cutoff, then
            the effect is considered significant)
        estimator : None or sklearn classifier object
            Specifies the classifier.
        n_folds : int
            number of folds for the cross-validation
        ignore_names : drug names to ignore in the comparisons (in addition to control)

    return:
        signif_effect_drugs : list
        low_effect_drugs : list
        mah_dist : dictionary
            A dictionary with the mahalanobis distances per dose for every
            drug name
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, recall_score
    from sklearn.model_selection import StratifiedKFold
    import pdb

    drug_name = np.array(drug_name)
    drug_dose = np.array(drug_dose)

    if ignore_names is None:
        ignore_names = []
    ignore_names.append(control)

    if estimator is None:
        estimator = LogisticRegression(penalty='l1', C=10, solver='liblinear', max_iter=500)

    cv_split = StratifiedKFold(n_splits=n_folds)

    if metric == 'recall':
        scorer = recall_score
    elif metric == 'mcc':
        scorer = matthews_corrcoef
    elif metric == 'balanced_accuracy':
        scorer = balanced_accuracy_score

    drug_names = np.unique(drug_name)

    scores = {}
    signif_effect_drugs = []
    for idr, drug in enumerate(drug_names):
        if drug in ignore_names:
            continue

        print('Checking compound {} ({}/{})...'.format(
            drug, idr+1, drug_names.shape[0]-len(ignore_names)))

        doses = np.sort(np.unique(drug_dose[drug_name==drug]))
        _iscores = []
        _iscores_shuffled = []
        _iscores_random = []
        pvals = []
        for dose in doses:
            if feat[(drug_name==drug) & (drug_dose==dose)].shape[0]<n_folds:
                _iscores.append(np.nan)
                _iscores_shuffled.append(np.nan)
                _iscores_random.append(np.nan)
                pvals.append(np.nan)
                continue
            print('dose {}..'.format(dose))
            X = pd.concat([
                feat[(drug_name==drug) & (drug_dose==dose)],
                feat[drug_name==control]
                ], axis=0).values

            ncontrol = np.sum(drug_name==control)

            y = np.concatenate([ np.ones(X.shape[0]-ncontrol), np.zeros(ncontrol) ])

            _cvscores = []
            for train_index, test_index in cv_split.split(X, y):
                estimator.fit(X[train_index], y[train_index])
                y_pred = estimator.predict(X[test_index])
                _cvscores.append( scorer(y[test_index], y_pred) )
            _iscores.append( np.mean(_cvscores) )

            y_shuffled = np.random.permutation(y)
            _cvscores_shuffled = []
            for train_index, test_index in cv_split.split(X, y_shuffled):
                estimator.fit(X[train_index], y_shuffled[train_index])
                y_pred = estimator.predict(X[test_index])
                _cvscores_shuffled.append( scorer(y_shuffled[test_index], y_pred) )
            _iscores_shuffled.append( np.mean(_cvscores_shuffled) )

            randsc = []
            for i in range(10000):
                randsc.append(recall_score(y, np.random.permutation(y)))
            _iscores_random.append( np.mean(randsc) )

            pvals.append( sum(np.array(randsc)>=np.mean(_cvscores))/len(randsc) )


        if np.any([p<pval_thres for p in pvals if p is not np.nan]):
        #if np.any([s>cutoff for s in _iscores if s is not np.nan]):
            signif_effect_drugs.append(drug)

        _iscores = pd.DataFrame({
            'dose': doses, metric: _iscores,
            '_'.join([metric, 'shuffled']): _iscores_shuffled,
            '_'.join([metric, 'random']): _iscores_random,
            '_'.join([metric, 'pval']): pvals})
        scores[drug] = _iscores


    low_effect_drugs = drug_names[
        ~np.isin(drug_names, [signif_effect_drugs+ignore_names])]

    return signif_effect_drugs, low_effect_drugs, scores

#%% DEPRECATED
def compounds_with_low_effect_multivariate(
        feat, drug_name, drug_dose, control='DMSO', signif_level=0.05,
        cov_estimator = 'EmpiricalCov',
        ignore_names = None, ignore_deprecation=False
        ):
    """
    Remove drugs when all the doses of the drug are very close to DMSO.
    Whether a dose is very close to DMSO is checked using the Mahalanobis
    distance (MD) calculated based on the robust covariance estimate of the
    DMSO observations and assuming that the MD^2 of DMSO points follow a chi2
    distribution with n_feat degrees of freedom.
    param:
        feat : dataframe
            feature dataframe
        drug_name : array-like
            the drug name per row in feat
        drug_dose : array-like
            the drug dose per row in feat
        control : str
            the name of the control samples in the drug_name array
        signif_level = float
            Defines the significance level for the p-value of the hypothesis
            test for each drug dose based on the MD^2 distribution.
        cov_estimator : 'RobustCov' or 'EmpiricalCov'
            Specifies the method to estimate the covariance matrix.
        ignore_names : drug names to ignore in the comparisons (in addition to control)

    return:
        signif_effect_drugs : list
        low_effect_drugs : list
        mah_dist : dictionary
            A dictionary with the mahalanobis distances per dose for every
            drug name
    """
    from sklearn.covariance import MinCovDet, EmpiricalCovariance
    from scipy.stats import chi2
    from time import time

    if ignore_deprecation:
        warnings.warn('This function has been deprecated.')
        return

    if ignore_names is None:
        ignore_names = []
    ignore_names.append(control)

    if cov_estimator == 'RobustCov':
        estimator = MinCovDet()
    elif cov_estimator == 'EmpiricalCov':
        estimator = EmpiricalCovariance()

    print('Estimating covariance matrix...'); st_time=time()
    estimator.fit(feat[drug_name==control])
    print('Done in {:.2f}.'.format(time()-st_time))

    drug_names = np.unique(drug_name)

    mah_dist = {}
    signif_effect_drugs = []
    for idr, drug in enumerate(drug_names):
        if drug in ignore_names:
            continue

        print('Checking compound {} ({}/{})...'.format(
            drug, idr+1, drug_names.shape[0]))

        X = feat[drug_name==drug]
        X.insert(0, 'dose', drug_dose[drug_name==drug])

        X = X.groupby(by='dose').mean()

        md2 = estimator.mahalanobis(X)
        mah_dist[drug] = md2

        nft = feat.shape[1]

        # Compute the P-Values
        p_vals = 1 - chi2.cdf(md2, nft)

        # Extreme values with a significance level of p_value
        if any(p_vals < signif_level):
            signif_effect_drugs.append(drug)

    low_effect_drugs = drug_names[~np.isin(drug_names, signif_effect_drugs)]

    return signif_effect_drugs, low_effect_drugs, mah_dist

#%%
if __name__=="__main__":
    from pathlib import Path
    from time import time
    import matplotlib.pyplot as plt

    # Input directory
    root_in = Path('/Users/em812/Documents/Workspace/Drugs/StrainScreens/' +
                   'SyngentaN2/create_datasets_both_screens/all_data')
    root_in = root_in / 'filtered_align_blue=True_average_dose=False_feat=all'

    data_file = root_in/ 'features.csv'
    meta_file = root_in/ 'metadata.csv'

    feat = pd.read_csv(data_file)
    meta = pd.read_csv(meta_file)

    #%% Test multi drug functions
    # Get some of the drugs
    drug_names = meta.loc[meta['drug_type']!='DMSO', 'drug_type'].unique()
    np.random.permutation(drug_names)
    drug_names = np.append(drug_names[:5], 'DMSO')
    mask = meta['drug_type'].isin(drug_names)

    # Get some of the features
    feat_names = feat.sample(n=300, axis=1).columns.to_list()

    # Run with parallel drugs
    test = 'LMM'
    st_time = time()
    signif_effect_drugs, low_effect_drugs, error_drugs, reject, pvals = \
        compounds_with_low_effect_univariate(
            feat.loc[mask, feat_names], meta.loc[mask,'drug_type'],
            meta.loc[mask,'drug_dose'],
            random_effect=meta.loc[mask,'date_yyyymmdd'],
            control='DMSO', test=test, comparison_type='multiclass',
            multitest_method='fdr_by', fdr=0.05,
            n_jobs=-1, return_pvals=True)
    lmm_time = time()-st_time

    # plt.figure(figsize=(8,8))
    # (pvals.T).hist(bins=50)
    # plt.savefig('test_fig_filter_compounds_{}.png'.format(test))

    test = 'Kruskal'
    st_time = time()
    signif_effect_drugs_krus, low_effect_drugs_krus, error_drugs_krus, reject_krus, pvals_krus = \
        compounds_with_low_effect_univariate(
            feat.loc[mask, feat_names], meta.loc[mask,'drug_type'],
            meta.loc[mask,'drug_dose'],
            control='DMSO', test=test, comparison_type='multiclass',
            multitest_method='fdr_by', fdr=0.05,
            n_jobs=-1, return_pvals=True)
    kruskal_time = time()-st_time

    # plt.figure(figsize=(8,8))
    # (pvals_krus.T).hist(bins=50)
    # plt.savefig('test_fig_filter_compounds_{}.png'.format(test))
