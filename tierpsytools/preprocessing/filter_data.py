#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:13:43 2019

@author: em812
"""
import warnings
import numpy as np
import pandas as pd

def filter_n_skeletons(feat, meta, min_nskel_per_video=None, min_nskel_sum=None,
                       verbose=True):
    """
    Filter samples based on number of skeletons in the video.

    Parameters
    ----------
    feat : dataframe shape=(n_samples, n_features)
        features dataframe
    meta : dataframe shape=(n_samples, n_metadata_columns)
        metadata dataframe.
        Must contain 'n_skeletons' column if align_bluelight=False.
        Must contain 'n_skeletons_prestim', 'n_skeletons_bluelight' and
        'n_skeletons_poststim' columns if align_bluelight=True.
    min_nskel_per_video : int or None
        Minimum number of skeletons for each video to keep the sample.
        This means that if the sample contains info from many videos,
        every one of these videos needs to have at least this number of skeletons
        for the sample to be kept.
        If None, then  min_nskel_sum must be defined.
    min_nskel_sum: int or None
        Minimum total number of skeletons to keep the sample.
        This means that if the sample contains info from many videos,
        the sum of the number of skeletons from all the videos needs to be at
        least this number for the sample to be kept.
        If None, then  min_nskel_per_video must be defined.
    align_bluelight : bool, optional
        Whether the bluelight conditions are aligned in the feat, meta dataframes.
        The default is False.

    Returns
    -------
    feat : filtered dataframe
    meta : filtered metadata

    """
    if min_nskel_per_video is None and min_nskel_sum is None:
        raise ValueError('You must define at least one of the min_nskel_* parameters.')

    n_samples = feat.shape[0]

    skel_cols = [col for col in meta.columns if col.startswith('n_skeletons')]

    if min_nskel_per_video is not None:
        mask = (meta[skel_cols]>=min_nskel_per_video).all(axis=1)
        feat = feat[mask]
        meta = meta[mask]
    if min_nskel_sum is not None:
        mask = meta[skel_cols].sum(axis=1)>=min_nskel_sum
        feat = feat[mask]
        meta = meta[mask]

    if verbose:
        print('{} samples dropped based on n_skeletons.'.format(n_samples-feat.shape[0]))
    return feat, meta

def drop_ventrally_signed(feat):
    """
    Drop features that are ventrally signed.

    Parameters
    ----------
    feat : pandas dataframe
        A dataframe with features as columns and samples as rows..

    Returns
    -------
    feat : pandas dataframe
        The filtered features dataframe.

    """

    abs_feat = [ft for ft in feat.columns if 'abs' in ft]

    ventr_feat = [ft.replace('_abs','') for ft in abs_feat]

    feat = feat[feat.columns.difference(ventr_feat, sort=False)]

    return feat


def select_feat_set(
        features, tierpsy_set_name=None, path_to_set=None, feat_list=None,
        append_bluelight=False, append_to_names=None):
    """
    Keep only features in a given feature set or in one of the predefined
    tierpsy feature sets.

    Parameters
    ----------
    features : pandas dataframe
        A dataframe with features as columns and samples as rows.
    tierpsy_set_name : str or None (optional). Default is None.
        This parameters can take one of the following values, if the user
        wants to choose one of the predefined tierpsy feature sets:
            - tierpsy_8
            - tierpsy_16
            - tierpsy_256
            - tierpsy_2k
        If the user wants to define a custom feature set, then this parameter
        must remain None.
    path_to_set : str or None (optional). Default is None.
        The path to a csv file containing the feature names to select.
    feat_list : list (optional). Default is None.
        A list of feature names to select.
    append_bluelight : bool (optional). Default is False.
        If True, the suffixes '_prestim', '_poststim', '_bluelight' will
        be appended to each feature name in the feature set. To be used
        when the features dataframe has aligned bluelight conditions, while
        the feature names in the selected feature set do not have bluelight
        suffixes.
    append_to_names : list of strings (optional). Default is None.
        The user can define custom made suffixes, similar to the bluelight
        descriptors to be appended to the feature names.

    Returns
    -------
    features : pandas dataframe
        The features dataframe containing only the selected features.
    """
    from os.path import join
    from tierpsytools import AUX_FILES_DIR

    check = [p is None for p in [tierpsy_set_name, path_to_set, feat_list]]
    if all(check):
        raise ValueError('Must define the feature set to select giving the '+
                         'name of a tierpsy feature set, a path to a feature '+
                         'set file or a list of features.')
    elif sum(check)<2:
        raise ValueError('Cannot define more than one feature set. Choose '+
                         'one way to define the features set between only '+
                         'tierpsy_set_name, path_to_set and feat_list.')

    if tierpsy_set_name is not None:
        if tierpsy_set_name not in ['tierpsy_8', 'tierpsy_16', 'tierpsy_256', 'tierpsy_2k']:
            raise ValueError('Tierpsy feature set name not recognised.')

        filenames = {
            'tierpsy_8': 'tierpsy_8.csv',
            'tierpsy_16': 'tierpsy_16.csv',
            'tierpsy_256': 'tierpsy_256.csv',
            'tierpsy_2k': 'tierpsy_2k.csv'
            }

        set_file = join(AUX_FILES_DIR,'feat_sets',filenames[tierpsy_set_name])

        ft_set = pd.read_csv(set_file, header=None).loc[:,0].to_list()

    elif path_to_set is not None:
        ft_set = pd.read_csv(path_to_set, header=None)[0].to_list()

    elif feat_list is not None:
        ft_set = feat_list

    if append_bluelight:
        bluelight_conditions = ['prestim', 'bluelight', 'poststim']
        ft_set = ['_'.join([ft, blue])
                  for ft in ft_set for blue in bluelight_conditions]

    if append_to_names is not None:
        ft_set = ['_'.join([ft, suf])
                  for ft in ft_set for suf in append_to_names]

    check = [ft in features.columns for ft in ft_set]
    if not np.all(check):
        warnings.warn(
            'The features dataframe does not contain all the features in '+
            'the selected features set. \n'+
            'Only {} of the {} features '.format(np.sum(check), len(ft_set))+
            'exist in the dataframe.')
        ft_set = [ft for ft in ft_set if ft in features.columns]

    return features[ft_set]


def filter_nan_inf(feat, threshold, axis, verbose=True):
    """
    FILTER_NAN_INF: function to remove features or samples based on the
    ratio of NaN+Inf values.
    param:
        feat = feature dataframe or np array (rows = samples, columns = features)
        threshold = max allowed ratio of NaN values within a feature or a sample
        axis = axis of filtering (0 --> filter features, 1 --> filter samples)
    return:
        feat = filtered feature matrix
    """
    import numpy as np

    sn = [(feat.shape[0], 'samples'), (feat.shape[1], 'features')]

    nanRatio = np.sum(np.logical_or(np.isnan(feat), np.isinf(feat)),
                      axis=axis) / np.size(feat, axis=axis)
    if axis==0:
        feat = feat.loc[:, nanRatio<threshold]
    else:
        feat = feat.loc[nanRatio<threshold,:]

    sn = [(s-feat.shape[i],n) for i,(s,n) in enumerate(sn)]

    if verbose:
        print('{} {} dropped.'.format(*sn[1-axis]))
    return feat


def cap_feat_values(feat, cutoff=1e15, remove_all_nan=True):
    """
    CAP_FEAT_VALUES: function to replace features
    with too large values (>cutoff) with the max value of the given
    feature in the remaining data points.
    param:
        feat = feature dataframe or np array containing features to be capped
               (rows = samples, columns = features)
        remove_all_nan = boolean (default=True). When remove_all_nan=True, a feature column
                that has only large values (>cutoff) will be removed from the feat matrix.
    return:
        feat = filtered feature matrix
    """
    isarray = False

    if isinstance(feat,np.ndarray):
        isarray = True
        feat = pd.DataFrame(feat)

    drop_cols = []
    for col in feat.columns:
        mask = ~feat[col].isna()
        ft = feat.loc[mask, col]
        if (ft>cutoff).all():
            drop_cols.append(col)
        else:
            maxvalid = ft[ft<=cutoff].max()
            ft[ft>cutoff] = maxvalid
            feat.loc[mask, col] = ft

    if remove_all_nan:
        feat = feat.drop(columns=drop_cols)

    if isarray:
        feat = feat.values

    return feat


def feat_filter_std(feat, threshold=0.0):
    """
    Remove features with std lower than a threshold
    param:
        feat = data frame or np array with features (rows = samples, columns = features)
        threshold = std value threshold. Any feature with std value equal or
        smaller to thies threshold will be dropped.
        If 0.0, then only features with zero std will be dropped.
    return:
        feat = filtered feature matrix
    """

    isarray = False
    if isinstance(feat,np.ndarray):
        isarray = True
        feat = pd.DataFrame(feat)

    feat = feat.loc[:,feat.std()>threshold]

    if isarray:
        feat = feat.values

    return feat

def drop_feat_by_keyword(feat, keywords):
    """
    Remove features that contain a keyword.
    param:
        feat = dataframe features (rows = samples, columns = features)
    return:
        feat = filtered feature matrix
    """
    import numpy as np

    if isinstance(keywords,(list,np.ndarray)):
        for key in keywords:
            feat = feat[feat.columns.drop(feat.filter(like=key))]
    elif isinstance(keywords,str):
        feat = feat[feat.columns.drop(feat.filter(like=keywords))]

    return feat

def select_feat_by_keyword(feat, keywords):
    """
    Select features that contain a keyword.
    param:
        feat = dataframe features (rows = samples, columns = features)
    return:
        feat = filtered feature matrix
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    feat = feat[[ft for ft in feat.columns if any([k in ft for k in keywords])]]

    return feat

def drop_samples_by_meta_column(
        feat, meta, column='drug_type', drop=['No', 'NoCompound'],
        verbose=False,):
    """
    Drop samples that correspond to specific categories in one of the metadata
    columns. The data type in meta[column] must be categorical or ordinal.
    For filtering based on a meta[column] with continuous values, use
    filter_samples_by_meta_col_thresholds instead.

    Parameters
    ----------
    feat : pandas dataframe
       Features matrix (rows = samples, columns = features)
    meta : pandas dataframe
        Metadata dataframe with row-to-row correspondance with the features
        dataframe. The dataframes feat and meta must have the same index.
    column : str
        One of the columns in the meta dataframe.
    drop : list or None.
        The categories in meta[column] to drop.
    verbose : bool, optional. The default is False.
        If True, a message with the number of samples dropped will be printed.

    Returns
    -------
    feat : pandas dataframe
       Filtered features matrix (rows = samples, columns = features)
    meta : pandas dataframe
        filtered metadata.

    """
    ids = meta[column].isin(drop)

    if verbose:
        print('Removing based on {}...:'.format(column))
        print('removed {} samples: '.format(sum(ids)))

    return feat[~ids], meta[~ids]

def select_samples_by_meta_column(
        feat, meta, column='drug_type', select=['No', 'NoCompound'],
        verbose=False,):
    """
    Select samples that correspond to specific values in one of the metadata
    columns. The data type in meta[column] must be categorical or ordinal.
    For filtering based on a meta[column] with continuous values, use
    filter_samples_by_meta_col_thresholds instead.

    Parameters
    ----------
    feat : pandas dataframe
       Features matrix (rows = samples, columns = features)
    meta : pandas dataframe
        Metadata dataframe with row-to-row correspondance with the features
        dataframe. The dataframes feat and meta must have the same index.
    column : str
        One of the columns in the meta dataframe.
    drop : list
        The categories in meta[column] to select.
    verbose : bool, optional. The default is False.
        If True, a message with the number of samples selected will be printed.

    Returns
    -------
    feat : pandas dataframe
       Filtered features matrix (rows = samples, columns = features)
    meta : pandas dataframe
        filtered metadata.

    """

    ids = meta[column].isin(select)

    if verbose:
        print('Selecting based on {}...:'.format(column))
        print('selected {} samples: '.format(feat.shape[0]))

    return feat[ids], meta[ids]


def filter_samples_by_meta_col_thresholds(
        feat, meta, column, min_value=None, max_value=None, verbose=False):
    """
    Filter samples based on value thresholds of a metadata columns with
    continuous data (for example n_skeletons).

    Parameters
    ----------
    feat : pandas dataframe
       Features matrix (rows = samples, columns = features)
    meta : pandas dataframe
        Metadata dataframe with row-to-row correspondance with the features
    column : str
        One of the columns in the meta dataframe. The data in meta[column]
        must be continuous or ordinal.
    min_value : int or float, optional. The default is None.
        The minimum value in meta[columns] to keep. If None, no min threshold
        will be applied.
    min_value : int or float, optional. The default is None.
        The maximum value in meta[columns] to keep. If None, no max threshold
        will be applied.
    verbose : bool, optional. The default is False.
        If True, a message with the number of samples selected will be printed.

    Returns
    -------
    feat : pandas dataframe
       Filtered features matrix (rows = samples, columns = features)
    meta : pandas dataframe
        filtered metadata.

    """
    ids = np.ones(feat.shape[0]).astype(bool)

    if min_value is not None:
        ids = ids & (meta[column].values >= min_value)
    if max_value is not None:
        ids = ids & (meta[column].values <= max_value)

    if verbose:
        print('Filtering based on {}...:'.format(column))
        print('removed {} samples: '.format(sum(~ids)))

    return feat[ids], meta[ids]


def drop_bad_wells(feat, meta, bad_well_cols=None, verbose=False):
    """
    Drop bad wells from the dataset, based on bool metadata columns that
    contain info about bad wells.

    Parameters
    ----------
    feat : pandas dataframe
       Features matrix (rows = samples, columns = features)
    meta : pandas dataframe
        Metadata dataframe with row-to-row correspondance with the features
        dataframe. The dataframes feat and meta must have the same index.
    bad_well_cols : list or None, optional. The default is None.
        The columns in the meta dataframe that contain information about bad wells.
        All the columns must contain bool data, with True indicating bad well.
        If None, then the function uses the columns in meta that contain 'is_bad'
        in their name.
    verbose : bool, optional. The default is False.
        If True, a message with the number of samples dropped will be printed.

    Returns
    -------
    feat : pandas dataframe
        Filtered feature df.
    meta : pandas dataframe
        Filtered metadata df.

    """

    # remove bad wells of any type
    n_samples = meta.shape[0]

    if bad_well_cols is None:
        bad_well_cols = [col for col in meta.columns if 'is_bad' in col]

    bad = meta[bad_well_cols].any(axis=1)

    if feat is not None:
        feat = feat.loc[~bad,:]
    meta = meta.loc[~bad,:]

    print('Bad wells removed: ', n_samples - meta.shape[0])

    return feat, meta


def select_bluelight_conditions(feat, meta, align_bluelight, select_bluelight):
    """
    For datasets that contain tierpsy features obtained for different bluelight
    conditions.
    Selects only the data that correspond to specific bluelight coditions.

    Parameters
    ----------
    feat : pandas dataframe
       Features matrix (rows = samples, columns = features)
    meta : pandas dataframe
        Metadata dataframe with row-to-row correspondance with the features
        dataframe. The dataframes feat and meta must have the same index.
    align_bluelight : bool
        If True, the features from the different bluelight conditions are aligned
        along axis=1 in the feat dataframe (wide format).
        If False, the features from the different blueligth conditions are
        stacked along axis=0. In this case, the meta dataframe must contain
        a column named 'bluelight' that indicated the blueligth condition of
        the corresponding row in the feat dataframe.
    select_bluelight : list
        The list of bluelight conditions to select.

    Raises
    ------
    ValueError
        If the parameter select_bluelight is not a list as expected.

    Returns
    -------
    feat : pandas dataframe
        The filtered features dataframe.
    meta : pandas dataframe
        The filtered metadata dataframe.

    """

    # Check input
    if not isinstance(select_bluelight, list):
        raise ValueError('choose_bluelight must be a list.')

    # Select bluelight conditions for long/wide df format
    if align_bluelight:
        drop_bluelight = set(['prestim','bluelight','poststim']).difference(set(select_bluelight))
        meta = meta[[col for col in meta.columns if not any([blue in col for blue in drop_bluelight]) ]]
        feat = feat[[col for col in feat.columns if not any([blue in col for blue in drop_bluelight]) ]]
    else:
        meta = meta[meta['bluelight'].isin(select_bluelight)]
        feat = feat.loc[meta.index, :]

    return feat, meta


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

