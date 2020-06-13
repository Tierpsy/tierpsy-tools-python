#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:28:09 2020

@author: em812
"""
import numpy as np

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

def remove_drugs_with_low_effect_univariate(
        feat, meta,
        threshold=0.05, fdr=0.05, test_each_dose=False,
        keep_names=['DMSO', 'NoCompound'],
        drugname_column = 'drug_type', drugdose_column = 'drug_dose'
        ):
    """
    Remove drugs when the number of features significantly different to DMSO
    for any dose is lower than the threshold.
    The statistical significance of the difference between a compound dose and
    the DMSO is assessed based on individual ANOVA tests for each feature.
    The Benjamini-Hochberg method is used to control the false discovery rate.
    param:
        feat : dataframe
            feature dataframe
        meta : dtaframe
            dataframe with metadata contraining a drugname_column and a
            drugdose_column
        threshold : float < 1.0 and < 0.0
            percentage of significant features detected to consider that the
            compound has significant effect
        fdr : float < 1.0 and > 0.0
            false discovery rate parameter in Benjamini-Hochberg method
        test_each_dose : bool, optional
            If true, each dose of each drug is tested for statistical
            significance compared to DMSO, and the drug is considered to
            have a significant effect if any of the doses satisfies the
            conditions set by the fdr and threshold parameters.
            If False, an ANOVA test is performed comparing the DMSO with all
            the doses (as separate classes) and the conditions are checked once
            for each drug.
        keep_names : list or None, optional
            list of names from the drugname_column to keep without checking
            for significance
        return_nonsignificant : bool, optional
            return the names of the drugs that are removed from the
            dataset
        drugname_column : string
            the name of the column in meta that contains the individual
            compound names
        drugdose_column : string
            the name of the column in meta that contains the drug doses
    return:
        signif_effect_drugs : list
        low_effect_drugs : list
        significant : dictionary
            mask of significant features for each drug name
    """
    from sklearn.feature_selection import SelectFdr, f_classif
    import pdb
    from statsmodels.stats.multitest import multipletests

    if meta[drugname_column].isna().any():
        raise ValueError('Drug names contain nan values.')

    if keep_names is None:
        keep_names = []

    #n_feat = feat.shape[1]
    drug_names = meta.loc[
        ~meta[drugname_column].isin(keep_names), drugname_column].unique()

    pvals = []
    for idrug,drug in enumerate(drug_names):
        print('Univariate tests for compound {}...'.format(drug))

        # For each dose get significant features using Benjamini-Hochberg
        # method with FDR=fdr
        X = feat[meta[drugname_column].isin([drug,'DMSO'])]
        y = meta.loc[meta[drugname_column].isin([drug,'DMSO']), drugdose_column]

        selector = SelectFdr(score_func=f_classif, alpha=fdr)

        if not test_each_dose:
            try:
                selector.fit(X, y)
            except ValueError:
                pdb.set_trace()

            pvals.append(np.asarray(selector.pvalues_))

        else:
            _pvals=[]
            for idose, dose in enumerate(y.unique()):
                if dose == 0:
                    continue
                selector.fit(X[np.isin(y,[0,dose])], y[np.isin(y,[0,dose])])

                _pvals.append(np.asarray(selector.pvalues_))

            pvals.append(_pvals)

    try:
        significant,_,_,_ = multipletests(
            np.vstack(pvals).reshape(-1), alpha=fdr, method='fdr_bh',
            is_sorted=False, returnsorted=False
            )
    except:
        pdb.set_trace()

    if not test_each_dose:
        significant = significant.reshape(np.array(pvals).shape)
        signif_effect_drugs = np.any(significant, axis=1)
    else:
        significant = significant.reshape(-1, X.shape[1])
        ndose = [len(x) for x in pvals]
        significant = np.split(significant, np.cumsum(ndose))[:-1]
        signif_effect_drugs = np.any([np.any(x, axis=0) for x in significant], axis=1)

    signif_effect_drugs = np.append(drug_names[signif_effect_drugs], keep_names)
    low_effect_drugs = drug_names[~np.isin(drug_names, signif_effect_drugs)]

    significant = {drug:mask for drug,mask in zip(drug_names, significant)}

    return signif_effect_drugs, low_effect_drugs, significant

def remove_drugs_with_low_effect_multivariate(
        feat, meta, signif_level=0.05,
        cov_estimator = 'EmpiricalCov',
        drugname_column = 'drug_type',
        dose_column = 'drug_dose',
        keep_names = ['DMSO', 'NoCompound']
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
        meta : dataframe
            dataframe with sample identification data
        signif_level = float
            Defines the significance level for the p-value of the hypothesis
            test for each drug dose based on the MD^2 distribution.
        cov_estimator : 'RobustCov' or 'EmpiricalCov'
            Specifies the method to estimate the covariance matrix.

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

    if meta[drugname_column].isna().any():
        raise ValueError('Drug names contain nan values.')

    if cov_estimator == 'RobustCov':
        estimator = MinCovDet()
    elif cov_estimator == 'EmpiricalCov':
        estimator = EmpiricalCovariance()

    print('Estimating covariance matrix...'); st_time=time()
    estimator.fit(feat[meta[drugname_column].isin(['DMSO'])])
    print('Done in {:.2f}.'.format(time()-st_time))

    drug_names = meta[drugname_column].unique()

    mah_dist = {}
    signif_effect_drugs = []
    for idr, drug in enumerate(drug_names):
        if drug in keep_names:
            continue

        print('Checking compound {} ({}/{})...'.format(
            drug, idr+1, drug_names.shape[0]))

        X = feat[meta[drugname_column].isin([drug])]
        X.insert(0, 'dose',
                 meta.loc[meta[drugname_column].isin([drug]), dose_column])

        X = X.groupby(by='dose').mean()

        md2 = estimator.mahalanobis(X)
        mah_dist[drug] = md2

        nft = feat.shape[1]

        # Compute the P-Values
        p_vals = 1 - chi2.cdf(md2, nft)

        # Extreme values with a significance level of p_value
        if any(p_vals < signif_level):
            signif_effect_drugs.append(drug)

    signif_effect_drugs.extend(keep_names)
    low_effect_drugs = drug_names[~np.isin(drug_names, signif_effect_drugs)]

    return signif_effect_drugs, low_effect_drugs, mah_dist


