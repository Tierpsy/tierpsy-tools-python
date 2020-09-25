#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:28:09 2020

@author: em812
"""
import numpy as np
import pandas as pd

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

def compounds_with_low_effect_univariate(
        feat, drug_name, drug_dose, control='DMSO',
        fdr=0.05, test='ANOVA', comparison_type='multiclass',
        ignore_names=['NoCompound'],
        return_pvals=False
        ):
    """
    Remove drugs when the number of features significantly different to DMSO
    for any dose is lower than the threshold.
    The statistical significance of the difference between a compound dose and
    the DMSO is assessed based on individual statistical tests for each feature
    (option for parametric and non-parameteric tests).
    The Benjamini-Hochberg method is used to control the false discovery rate.
    param:
        feat : dataframe
            feature dataframe, shape=(n_samples, n_features)
        drug_name : array, shape=(n_samples)
            defines the type of drug in each sample
        drug_dose : array, shape=(n_samples)
            defines the drug dose in each sample (expects 0 dose for controls)
        control : str
            the name of the control samples in the drug_name array
        fdr : float < 1.0 and > 0.0
            false discovery rate parameter in Benjamini-Hochberg method
        test : str, options: ['ANOVA', 'Kruskal_Wallis', 'Wilkoxon_Rank_Sum']
            The type of statistical test to perform for each feature.
        comparison_type : str, options: ['binary', 'multiclass', 'binary_each_dose']
            defines the groups seen in the statistical test.
            If 'binary', then the controls are compared to all the drug samples
            pooled together.
            If 'multiclass', then the controls and each drug dose are all
            considered separate groups.
            If 'binary_each_dose', then n_doses tests are performed for each
            feture. Each test compares one dose to the controls. If any of the
            tests reaches the significance thresshold, then the feature
            is considered significant.
        ignore_names : list or None, optional
            list of names from the drug_name array to ignore in the comparisons
            (in addition to the control)
        return_pvals: bool
            Wether or not to return the pvalues of all the comparisons
    return:
        signif_effect_drugs : list
        low_effect_drugs : list
        significant : dictionary
            mask of significant features for each drug name
    """
    from scipy.stats import kruskal, ranksums, f_oneway
    import pdb
    from statsmodels.stats.multitest import multipletests
    from functools import partial

    # Ignore the control and any names defined by user
    if ignore_names is None:
        ignore_names = []
    ignore_names.append(control)

    # Local function for parallel processing of univariate tests for each drug
    def stats_test(X, y, test, **kwargs):
        from joblib import Parallel, delayed

        def _one_feat(samples, **kwargs):
            samples = [s[~np.isnan(s)] for s in samples if not all(np.isnan(s))]
            if len(samples)<2:
                return (np.nan, np.nan)
            return test(*samples, **kwargs)

        parallel = Parallel(n_jobs=-1, verbose=True)
        func = delayed(_one_feat)

        res = parallel(
            func([sample[:,ix]
                  for sample in [np.array(X[y==iy]) for iy in np.unique(y)]],
                 **kwargs)
            for ix in range(X.shape[1]))

        ss = [s for s,p in res]
        ps = [p for s,p in res]

        return ss, ps

    # Create the function that will test every feature of a given drug
    if test == 'ANOVA':
        func = partial(stats_test, test=f_oneway)
    elif test.startswith('Kruskal'):
        func = partial(stats_test, test=kruskal, nan_policy='raise')
    elif test.startswith('Wilkoxon'):
        if comparison_type=='multiclass':
            raise ValueError(
                'The Wilkoxon rank sum test can not be used with the multiclass comparison type, '+\
                'as it can only be used for comparison between two samples.')
        func = partial(stats_test, test=ranksums)

    # Get the list of drug names to test
    drug_names = np.array([drug for drug in np.unique(drug_name) if drug not in ignore_names])

    # Run the univariate tests for every drug in drug_names
    pvals = []
    significant = []
    for idrug,drug in enumerate(drug_names):
        print('Univariate tests for compound {}...'.format(drug))

        # For each dose get significant features using Benjamini-Hochberg
        # method with FDR=fdr
        X = feat[np.isin(drug_name, [drug, control])]
        y = drug_dose[np.isin(drug_name, [drug, control])]
        if comparison_type=='binary':
            y[y>0] = 1

        if comparison_type=='multiclass' or comparison_type=='binary':
            try:
                _, c_pvals = func(X, y)
            except ValueError:
                pdb.set_trace()

            c_sign, c_pvals,_,_ = multipletests(
                c_pvals, alpha=fdr, method='fdr_bh',
                is_sorted=False, returnsorted=False
                )
            pvals.append(c_pvals)
            significant.append(c_sign)

        elif comparison_type=='binary_each_dose':
            c_pvals=[]
            for idose, dose in enumerate(np.unique(y)):
                if dose == 0:
                    continue
                _, _pvals = func(X[np.isin(y,[0,dose])], y[np.isin(y,[0,dose])])

                c_pvals.append(_pvals)

            _shape = np.array(c_pvals).shape
            c_sign, c_pvals,_,_ = multipletests(
                np.vstack(c_pvals).reshape(-1), alpha=fdr, method='fdr_bh',
                is_sorted=False, returnsorted=False
                )
            c_pvals = c_pvals.reshape(_shape)
            c_sign = c_sign.reshape(_shape)
            pvals.append(c_pvals)
            significant.append(c_sign)

        else:
            raise ValueError('Comparison type not recognised.')

    if comparison_type=='binary' or comparison_type=='multiclass':
        error = np.all(np.isnan(pvals), axis=1)
        signif_effect = np.any(np.array(significant), axis=1)
    else:
        error = np.array([np.all(np.isnan(x)) for x in pvals])
        signif_effect = np.any([np.any(x, axis=0) for x in significant], axis=1)

    error_drugs = drug_names[error]
    signif_effect_drugs = drug_names[signif_effect & ~error]
    low_effect_drugs = drug_names[~error & ~signif_effect]

    significant = {drug:mask for drug,mask in zip(drug_names, significant)}
    pvals = {drug:mask for drug,mask in zip(drug_names, pvals)}

    if return_pvals:
        return signif_effect_drugs, low_effect_drugs, error_drugs, significant, pvals
    else:
        return signif_effect_drugs, low_effect_drugs, error_drugs

def compounds_with_low_effect_multivariate(
        feat, drug_name, drug_dose, control='DMSO', signif_level=0.05,
        cov_estimator = 'EmpiricalCov',
        ignore_names = None
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

