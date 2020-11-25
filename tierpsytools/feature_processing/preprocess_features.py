#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:13:43 2019

@author: em812
"""
import warnings
import numpy as np
import pandas as pd


def impute_nan_inf(feat, groupby=None):
    """
    IMPUTE_NAN_INF: replace NaN and inf values with feature average
    param:
        feat : dataframe or np array
            Features matrix (rows = samples, columns = features)
        groupby : array or list of arrays
            Ids based on which the feat dataframe will be grouped. The nans
            will be imputed with the mean values of each group independently.
    return:
        feat = feature matrix without nan/inf
    """

    isarray = False
    if isinstance(feat, np.ndarray):
        isarray = True
        feat = pd.DataFrame(feat)

    # replace inf with nan
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # fill in nans with mean values of cv features for each strain separately
    if groupby is None:
        feat = feat.fillna(feat.mean())
    else:
        feat = [x for _,x in feat.groupby(by=groupby, sort=True)]
        for i in range(len(feat)):
            feat[i] = feat[i].fillna(feat[i].mean())
        feat = pd.concat(feat, axis=0).sort_index()

    # Covert back to array
    if isarray:
        feat = feat.values

    return feat


def average_by_groups(
        feat, meta,
        groupby=['worm_strain', 'drug_type', 'imaging_plate_drug_concentration'],
        align_bluelight=None):
    """
    Get average feature values within selected groups and create a matching
    metadata dataframe to the averaged features dataframe.

    Parameters
    ----------
    feat : pandas dataframe
       Features matrix (rows = samples, columns = features)
    meta : pandas dataframe
        Metadata dataframe with row-to-row correspondance with the features
        dataframe. The dataframes feat and meta must have the same index.
    groupby : list, optional. The default is ['worm_strain', 'drug_type',
                                              'imaging_plate_drug_concentration'].
        A list of columns in the meta dataframe that define the groups within
        which features will be averaged.
    align_bluelight : bool or None, optional. Default is None.
        If None (no bluelight stimulus in the dataset) or True (bluelight
        conditions aligned along axis=1 in the features dataframe), then the
        averaging is done only based on the groupby columns.
        If False (bluelight conditions stacked aling axis=0 and 'bluelight'
        column exists in the meta dataframe), then the 'bluelight' column is
        added to the groupby list, so that the averaging will happen across the
        same bluelight condition.

    Returns
    -------
    feat : pandas dataframe
        The features matrix with averaged values.
    meta : pandas dataframe
        The metadata dataframe corresponding to the averaged features dataframe.

    """

    if align_bluelight is not None:
        if not align_bluelight:
            groupby.extend(['bluelight'])

    feat = feat.groupby(by=[meta[key] for key in groupby]).mean()
    meta = meta.drop_duplicates(subset=groupby)
    meta = meta.set_index(groupby).loc[feat.index]

    meta = meta.reset_index(drop=False)
    feat = feat.reset_index(drop=True)

    return feat, meta


def encode_categorical_variable(feat, variable, base_name=None, encoder=None):
    """
    Encode a categorical variable and add it to the features dataframe.

    Parameters
    ----------
    feat : pandas dataframe
       Features matrix (rows = samples, columns = features)
    variable : array, list or pandas series
        The categorical variable to encode. Must have the same length as the
        number of rows in the feat dataframe.
    base_name : str, optional. The default is None.
        If defined, it will be used to name the encoded features.
    encoder : encoder instance, optional. The default is None.
        An encoder class instance, with the sklearn format (must have a fit_transform method).
        If None, then the OneHotEncoder of sklearn will be used.

    Returns
    -------
    feat : pandas dataframe
        features matrix including the encoded features.

    """

    from sklearn.preprocessing import OneHotEncoder

    if encoder is None:
        encoder = OneHotEncoder(sparse=False)
    if isinstance(variable, pd.Series):
        if base_name is None:
            base_name = variable.name
        variable = variable.values

    encoded_ft = encoder.fit_transform(variable.reshape(-1,1))
    if len(encoded_ft.shape)==1:
        if base_name is None:
            base_name = 'encoded_feature'
        feat.insert(0, base_name, encoded_ft)
    else:
        if base_name is None:
            names = encoder.categories_[0]
        else:
            names = ['_'.join([base_name, ctg]) for ctg in encoder.categories_[0]]
        for col, names in enumerate(names):
            feat.insert(0, names, encoded_ft[:, col])
    return feat

