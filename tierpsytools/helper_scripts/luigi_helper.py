#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:27:43 2020
@author: lferiani
"""

import scipy
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from tierpsytools.read_data.get_timeseries import read_timeseries


ID_COLS = ['date_yyyymmdd',
           'imaging_plate_id',
           'well_name']

CATEG_COLS = ID_COLS + ['worm_gene', 'worm_strain']

COLS_MOTION_CHANGE = ['fw2bw', 'bw2fw',  # diff is 2 or -2
                      'bw2st', 'st2fw',  # diff is 1, motion 0 or 1
                      'fw2st', 'st2bw']  # diff is -1, motion 0 or -1

COLS_MOTION_MODES = ['is_fw', 'is_bw', 'is_st', 'is_nan']

HIRES_COLS = ['worm_index', 'timestamp', 'speed', 'd_speed',
              'length', 'width_head_base', 'width_midbody', 'd_speed_midbody',
              'motion_mode']

MD_COLS = ['date_yyyymmdd', 'imaging_plate_id', 'well_name',
           'worm_strain', 'drug_type',
           'imaging_plate_drug_concentration',
           'imaging_plate_drug_concentration_units',
           'imgstore_name_pre',
           'imgstore_name_blue',
           'imgstore_name_post']

FORCED_FLOAT = ['time_binned_s', 'index', 'timestamp', 'time_s']

def read_metadata(metadata_fname, query_str=None, exclude=None,
                  only_useful_cols=True):

    md_df = pd.read_csv(metadata_fname)
    if query_str is not None:
        md_df.query(query_str, inplace=True)

    # reshape the dataframe slightly so we only have one line per (plate,well)
    pre_df = md_df[md_df['imgstore_name'].str.contains('prestim')]
    blue_df = md_df[md_df['imgstore_name'].str.contains('bluelight')]
    post_df = md_df[md_df['imgstore_name'].str.contains('poststim')]

    cols_to_join = list(set(pre_df.columns) - set(['imgstore_name']))

    md_df = pd.merge(pre_df, blue_df,
                     how='outer',
                     on=cols_to_join,
                     suffixes=('_pre', '_blue'))
    md_df = pd.merge(md_df, post_df,
                     how='outer',
                     on=cols_to_join,
                     suffixes=('', '_post'))
    md_df.rename(columns={'imgstore_name': 'imgstore_name_post'},
                 inplace=True)

    # if necessary, filter according to the files passed
    if exclude is not None:
        idx_todrop = pd.concat(
            [md_df[col].apply(lambda x: Path(x).stem in exclude)
             for col in ['imgstore_name_blue',
                         'imgstore_name_pre',
                         'imgstore_name_post']], axis=1).any(axis=1)

        # drop the unwanted lines
        md_df = md_df[np.logical_not(idx_todrop)]

    if only_useful_cols is True:
        keep_cols = md_df.columns.intersection(MD_COLS)
        md_df = md_df[keep_cols]

    if 'drug_type' not in md_df.columns:
        warnings.warn('No drug_type column in metadata')
    else:
        idx_nan_drugtype = md_df['drug_type'].isna()
        if idx_nan_drugtype.any():
            print('metadata contains nan in "drug_type" here:')
            print(md_df[idx_nan_drugtype][['date_yyyymmdd',
                                           'imaging_plate_id',
                                           'well_name']])
            print('Dropping nan drug_type')
            md_df = md_df[~idx_nan_drugtype]

        # check that drug concentration is not empty in DMSO, NoCompound
        # if empty, fill it
        for ctrl in ['DMSO', 'NoCompound']:
            idx_nan = ((md_df['drug_type'] == ctrl) &
                       md_df['imaging_plate_drug_concentration'].isna())
            if idx_nan.sum() > 0:
                print('nans in {} concentration, imputing'.format(ctrl))
                md_df.loc[idx_nan, 'imaging_plate_drug_concentration'] = (
                    get_value_from_const_column(
                        md_df.query('drug_type=="{}"'.format(ctrl)),
                        'imaging_plate_drug_concentration'))

    if md_df.isna().any().any():
        warnings.warn('There are still nans in the metadata!')

    md_df['date_yyyymmdd'] = md_df['date_yyyymmdd'].astype(str)

    return md_df


def downsample_timeseries(df, fps=25, time_bin_s=1):
    # import pdb; pdb.set_trace()
    # convert to seconds
    df['time_s'] = df['timestamp'] / fps
    # bin time. time is floored to the left of the bin
    df['time_binned_s'] = df['time_s'] // time_bin_s * time_bin_s
    # now time average within each worm. worm_index is unique for the video
    cols_to_group_by = intersect(df.columns, CATEG_COLS)
    cols_to_group_by.extend(['worm_index', 'time_binned_s'])
    df_g = df.groupby(cols_to_group_by, observed=True)
    dwnsmpl_df = df_g.mean()
    # different behaviour for motion_mode and number worms
    dwnsmpl_df.drop(columns=['motion_mode'], inplace=True)
    try:
        dwnsmpl_df['motion_mode'] = df_g['motion_mode'].agg(
            lambda x: scipy.stats.mode(x)[0])
    except ValueError as EE:
        print(EE)
        import pdb
        pdb.set_trace()
    dwnsmpl_df.reset_index(inplace=True)
    dwnsmpl_df['well_name'] = dwnsmpl_df['well_name'].astype('category')
    # also need to count how many worms per well per time bin
    # in each well/time_bin, per each timestamp, count how many times it occurs
    # then take the maximum number of occurrences of any timestamp.
    # that's the maximum number of contemporaneous worms in the well/time_bin
    # in other words, take the number of counts for the mode of the timestamps
    nworms_per_wellbin = df.groupby(
        ['well_name', 'time_binned_s'], observed=True)['timestamp'].agg(
            lambda x: scipy.stats.mode(x)[1])
    nworms_per_wellbin = nworms_per_wellbin.to_frame(
        name='n_worms/(well,time_bin)').reset_index()
    dwnsmpl_df = pd.merge(dwnsmpl_df, nworms_per_wellbin, how='left',
                          on=['well_name', 'time_binned_s'])
    # now do the same, without the well_name
    nworms_per_bin = df.groupby('time_binned_s')['timestamp'].agg(
            lambda x: scipy.stats.mode(x)[1])
    nworms_per_bin = nworms_per_bin.to_frame(
        name='n_worms/time_bin').reset_index()
    dwnsmpl_df = pd.merge(dwnsmpl_df, nworms_per_bin, how='left',
                          on=['time_binned_s'])
    return dwnsmpl_df


# def find_motion_changes(df):
#     # preallocate columns
#     for col in COLS_MOTION_CHANGE:
#         df[col] = False
#     # look at change in motion_mode on a worm-by-worm basis
#     df_g = df.groupby('worm_index')
#     for wi, worm in df_g:
#         assert (worm['timestamp'].diff().dropna() > 0).all, \
#             'df not sorted by time!'
#         # change in motion mode (row - earlier row)
#         change = worm['motion_mode'].diff()
#         # backwards to forwards
#         idx = worm[change == 2].index
#         df.loc[idx, 'bw2fw'] = True
#         # forwards to backwards
#         idx = worm[change == -2].index
#         df.loc[idx, 'fw2bw'] = True
#         # backwards to stationary
#         idx = worm[(change == 1) & (worm['motion_mode'] == 0)].index
#         df.loc[idx, 'bw2st'] = True
#         # stationary to forwards
#         idx = worm[(change == 1) & (worm['motion_mode'] == 1)].index
#         df.loc[idx, 'st2fw'] = True
#         # forwards to stationary
#         idx = worm[(change == -1) & (worm['motion_mode'] == 0)].index
#         df.loc[idx, 'fw2st'] = True
#         # stationary to backwards
#         idx = worm[(change == -1) & (worm['motion_mode'] == -1)].index
#         df.loc[idx, 'st2bw'] = True

#     # now clump together some motion changes for ease of plotting later
#     df['motion_up'] = df[['bw2fw', 'bw2st', 'st2fw']].any(axis=1)
#     df['motion_down'] = df[['fw2bw', 'st2bw', 'fw2st']].any(axis=1)
#     df['motion_change'] = df[COLS_MOTION_CHANGE].any(axis=1)

#     return df
def add_motion_mode_cols(df):
    if all([c in df.columns for c in COLS_MOTION_MODES]):
        print('motion_mode columns already present, skipping...')
        return df
    assert 'motion_mode' in df.columns, 'no motion_mode column in dataframe'
    df['is_bw'] = df['motion_mode'] == -1
    df['is_st'] = df['motion_mode'] == 0
    df['is_fw'] = df['motion_mode'] == 1
    df['is_nan'] = df['motion_mode'].isna()
    return df


def find_motion_changes(df):
    add_motion_mode_cols(df)
    if 'well_id' in df.columns:
        cols_to_group_by = ['well_id', 'worm_index']
    elif all([c in df.columns for c in CATEG_COLS]):
        cols_to_group_by = CATEG_COLS
    else:
        cols_to_group_by = ['well_name', 'worm_index']
    print(cols_to_group_by)
    df['motion_diff'] = df.groupby(
        cols_to_group_by,
        observed=True)['motion_mode'].transform(
            pd.Series.diff)
    # create extra temporary columns
    for col in COLS_MOTION_CHANGE:
        df[col] = False
    df['-1'] = False
    df['1'] = False
    df[['fw2bw', 'bw2fw', '-1', '1']] = (
        df[['motion_diff']*4].values == [-2, 2, -1, 1])
    df['fw2st'] = df[['-1', 'is_st']].all(axis=1)
    df['st2bw'] = df[['-1', 'is_bw']].all(axis=1)
    df['bw2st'] = df[['1', 'is_st']].all(axis=1)
    df['st2fw'] = df[['1', 'is_fw']].all(axis=1)
    df.drop(columns=['-1', '1'], inplace=True)
    # now clump together some motion changes for ease of plotting later
    df['motion_up'] = df[['bw2fw', 'bw2st', 'st2fw']].any(axis=1)
    df['motion_down'] = df[['fw2bw', 'st2bw', 'fw2st']].any(axis=1)
    df['motion_change'] = df[COLS_MOTION_CHANGE].any(axis=1)
    return df


def count_motion_modes(df):
    # copy all label columns plus timestamp and motion_mode
    cols_to_copy = get_nonnumorbool_cols(df)
    time_col = 'time_s' if 'time_s' in df.columns else 'timestamp'
    cols_to_copy = cols_to_copy + [time_col, 'motion_mode']
    if all([c in df.columns for c in COLS_MOTION_MODES]):
        cols_to_copy.extend(COLS_MOTION_MODES)

    foo = df[cols_to_copy].copy()
    if not all([c in df.columns for c in COLS_MOTION_MODES]):
        # add columns with hits when a motion mode matched the column name
        add_motion_mode_cols(foo)
        # equivalent of:
        # foo['is_bw'] = foo['motion_mode'] == -1
        # foo['is_st'] = foo['motion_mode'] == 0
        # foo['is_fw'] = foo['motion_mode'] == 1
        # foo['is_nan'] = foo['motion_mode'].isna()
    # add a dummy column to sum over at the end.
    # not incorrect anyways as n_worms per line of df is 1
    foo['n_worms'] = 1
    # select which columns to group by. We want info by well and frame
    cols_to_group_by = intersect(df.columns, CATEG_COLS)
    if 'well_id' in cols_to_copy:
        cols_to_group_by.append('well_id')
    cols_to_group_by.append(time_col)
    print(foo.columns)
    print(cols_to_group_by)
    # count motion_modes occurrencies
    motion_mode_counts = foo.groupby(cols_to_group_by,
                                     observed=True).sum()
    print(motion_mode_counts.columns)
    # motion mode is now superfluous
    motion_mode_counts.drop(columns='motion_mode', inplace=True)
    # now rename columns to reflect that you've summed
    coldict = {c: 'n_worms_' + c.split('_')[-1]
               for c in motion_mode_counts.columns
               if c != 'n_worms'}
    motion_mode_counts.rename(columns=coldict, inplace=True)

    return motion_mode_counts


def get_frac_motion_modes(df):
    """get_frac_motion_modes
    divide number of worms in a motion mode by
    the total number of worms in that frame
    """
    assert not any([c in df.columns.to_list() + df.index.names
                    for c in ['well_name', 'well_id']]), \
        'data not consolidated'
    frac_motion_modes = df.drop(columns='n_worms')
    frac_motion_modes = frac_motion_modes.divide(df['n_worms'], axis=0)
    print(frac_motion_modes.columns)
    # rename columns now
    frac_motion_modes.rename(lambda x: x.replace('n_worms_', 'frac_worms_'),
                             axis='columns',
                             inplace=True)
    return frac_motion_modes


def get_value_from_const_column(df, colname):
    assert df[colname].nunique() == 1, 'Non constant values or all nans!'
    return df[~df[colname].isna()][colname].iloc[0]


def plot_stimuli(ax=None, units='s', fps=25,
                 stimulus_start=[60, 160, 260],
                 stimulus_duration=10):
    """plot_stimuli
    plots patches at the times when the stimulus was supposed to be on.
    Stimuli starts and duration are hardcoded atm"""
    if ax is None:
        ax = plt.gca()

    if units == 'frames':
        stimulus_start = [s * fps for s in stimulus_start]
        stimulus_duration = stimulus_duration * fps

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    for ss in stimulus_start:
        rect = plt.Rectangle(xy=(ss, ymin),
                             width=stimulus_duration,
                             height=yrange,
                             alpha=0.1,
                             facecolor='tab:blue')
        ax.add_patch(rect)
    return


def short_plot_stimuli(ax=None, units='s', fps=25,
                 stimulus_start=[60],
                 stimulus_duration=10):
    """plot_stimuli
    plots patches at the times when the stimulus was supposed to be on.
    Stimuli starts and duration are hardcoded atm"""
    if ax is None:
        ax = plt.gca()

    if units == 'frames':
        stimulus_start = [s * fps for s in stimulus_start]
        stimulus_duration = stimulus_duration * fps

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    for ss in stimulus_start:
        rect = plt.Rectangle(xy=(ss, ymin),
                             width=stimulus_duration,
                             height=yrange,
                             alpha=0.1,
                             facecolor='tab:blue')
        ax.add_patch(rect)
    return


# def load_bluelight_timeseries_from_results(metadata_df, results_dir):

#     timeseries_df = []  # this will be the massive df
#     hires_df = []  # this will only have a few features
#     # so group metadata by ideo. so far, let's use the bluelight only
#     metadata_g = metadata_df.groupby('imgstore_name_blue')
#     # loop on video, select which wells need to be read
#     for gname, md_group in tqdm(metadata_g):
#         # checks and extract variables out
#         md_group[CATEG_COLS]
#         # plate_id = get_value_from_const_column(md_group, 'imaging_plate_id')
#         # date = get_value_from_const_column(md_group, 'date_yyyymmdd')
#         # worm_strain = get_value_from_const_column(md_group, 'worm_strain')
#         # what do we want to read?
#         filename = results_dir / gname / 'metadata_featuresN.hdf5'
#         wells_to_read = list(md_group['well_name'])
#         # read data
#         data = read_timeseries(filename, only_wells=wells_to_read)
#         # filter bad worm trajectories
#         data = filter_timeseries(data)
#         # add plate id and date and worm strain
#         data = pd.merge(data, md_group[CATEG_COLS], how='left', on='well_name')
#         data['date_yyyymmdd'] = data['date_yyyymmdd'].astype(str)
#         # data['imaging_plate_id'] = plate_id
#         # data['date_yyyymmdd'] = str(date)
#         # data['worm_strain'] = worm_strain
#         # extract the hires features
#         hires_data = data[CATEG_COLS + HIRES_COLS].copy()
#         # downsample the main df
#         data = downsample_timeseries(data, fps=25, time_bin_s=1)

#         # set some columns to categorical for memory saving
#         for col in CATEG_COLS:
#             data[col] = data[col].astype('category')
#             hires_data[col] = hires_data[col].astype('category')
#         # now grow list
#         hires_df.append(hires_data)
#         timeseries_df.append(data)

#     # join in massive dataframe
#     timeseries_df = pd.concat(timeseries_df, axis=0, ignore_index=True)
#     hires_df = pd.concat(hires_df, axis=0, ignore_index=True)

#     # create a single unique well id
#     hires_df['well_id'] = make_well_id(hires_df)
#     timeseries_df['well_id'] = make_well_id(timeseries_df)

#     # find motion changes
#     find_motion_changes(timeseries_df)
#     find_motion_changes(hires_df)

#     # fix types
#     categ_cols = CATEG_COLS + ['well_id']
#     for col in categ_cols:
#         timeseries_df[col] = timeseries_df[col].astype('category')
#         hires_df[col] = hires_df[col].astype('category')
#     # timeseries_df['motion_mode'] = timeseries_df['motion_mode'].astype(int)
#     # hires_df['motion_mode'] = hires_df['motion_mode'].astype(int)

#     return timeseries_df, hires_df


def load_bluelight_timeseries_from_results(
        metadata_df, results_dir, saveto=None):

    timeseries_df = []  # this will be the massive df
    hires_df = []  # this will only have a few features
    from pandas.api.types import CategoricalDtype

    # need to kknow in advance all possible values of "categorical" values.
    # just read them from the metadata!
    cat_types = {}
    for cat in CATEG_COLS:
        vals = metadata_df[cat].unique()
        cat_types[cat] = CategoricalDtype(categories=vals, ordered=False)
    vals = make_well_id(metadata_df).unique()
    cat_types['well_id'] = CategoricalDtype(categories=vals, ordered=False)

    # so group metadata by ideo. so far, let's use the bluelight only
    metadata_g = metadata_df.groupby('imgstore_name_blue')
    # loop on video, select which wells need to be read
    for gcounter, (gname, md_group) in tqdm(enumerate(metadata_g)):
        # checks and extract variables out
        
        md_group[CATEG_COLS]
        # plate_id = get_value_from_const_column(md_group, 'imaging_plate_id')
        # date = get_value_from_const_column(md_group, 'date_yyyymmdd')
        # worm_strain = get_value_from_const_column(md_group, 'worm_strain')
        # what do we want to read?
        filename = results_dir / gname / 'metadata_featuresN.hdf5'
        wells_to_read = list(md_group['well_name'])
        # read data
        data = read_timeseries(filename, only_wells=wells_to_read)
       
        # filter bad worm trajectories
        data = filter_timeseries(data)
        if data.empty:
            print('No data in {}, {}, {}'.format(gcounter, gname, md_group))
            continue
        # add plate id and date and worm strain
        data = pd.merge(data, md_group[CATEG_COLS], how='left', on='well_name')
        data['date_yyyymmdd'] = data['date_yyyymmdd'].astype(str)
        # data['imaging_plate_id'] = plate_id
        # data['date_yyyymmdd'] = str(date)
        # data['worm_strain'] = worm_strain
        # extract the hires features
        hires_data = data[CATEG_COLS + HIRES_COLS].copy()
        # downsample the main df
        # try:
            
        data = downsample_timeseries(data, fps=25, time_bin_s=1)
        #     continue
        # except Exception:
        #     print(gname, md_group, gcounter)
        # create a single unique well id
        hires_data['well_id'] = make_well_id(hires_data)
        data['well_id'] = make_well_id(data)

        # find motion changes
        find_motion_changes(data)
        find_motion_changes(hires_data)
        # import pdb; pdb.set_trace()
        # set some columns to categorical for memory saving
        for col in CATEG_COLS:
            data[col] = data[col].astype(cat_types[col])
            hires_data[col] = hires_data[col].astype(cat_types[col])

        for col in FORCED_FLOAT:
            data[col] = data[col].astype(float)
            # hires_data[col] = hires_data[col].astype(float[col])
            
        # for col in CATEG_COLS:
        #     data[col] = data[col].astype(cat_types[col])
        #     hires_data[col] = hires_data[col].astype(cat_types[col])

        if saveto is None:
            # now grow list
            hires_df.append(hires_data)
            timeseries_df.append(data)
        else:
            # save to disk
            is_append = False if (gcounter == 0) else True
            try:
                data.to_hdf(
                    saveto, 'timeseries_df', format='table', append=is_append)
            except: 
                data.dtypes.to_csv('~/Desktop/foo.csv')
                raise Exception()
                
            imaging_plate_id = get_value_from_const_column(md_group,
                                                           'imaging_plate_id')
            hires_data.to_hdf(
                saveto, 'hires_df_{}'.format(imaging_plate_id),
                format='table', append=True,
                min_itemsize={'well_id': 30},
                data_columns=True)

    if saveto is None:
        # join in massive dataframe
        timeseries_df = pd.concat(timeseries_df, axis=0, ignore_index=True)
        hires_df = pd.concat(hires_df, axis=0, ignore_index=True)
        # fix types
        categ_cols = CATEG_COLS + ['well_id']
        for col in categ_cols:
            timeseries_df[col] = timeseries_df[col].astype('category')
            hires_df[col] = hires_df[col].astype('category')
        # out
        return timeseries_df, hires_df
    else:
        return



def just_load_one_timeseries(metadata_df, results_dir, fileno=0):
    """only use for debugging really"""
    # so group metadata by video. so far, let's use the bluelight only
    metadata_g = metadata_df.groupby('imgstore_name_blue')
    gname = list(metadata_g.groups.keys())[fileno]
    md_group = metadata_g.get_group(gname)
    plate_id = get_value_from_const_column(md_group, 'imaging_plate_id')
    date = get_value_from_const_column(md_group, 'date_yyyymmdd')
    worm_strain = get_value_from_const_column(md_group, 'worm_strain')
    # what do we want to read?
    filename = results_dir / gname / 'metadata_featuresN.hdf5'
    wells_to_read = list(md_group['well_name'])
    # read data
    data = read_timeseries(filename, only_wells=wells_to_read)
    data['imaging_plate_id'] = plate_id
    data['date_yyyymmdd'] = str(date)
    data['worm_strain'] = worm_strain
    data['well_id'] = make_well_id(data)
    return data, plate_id, date


def filter_timeseries(ts):
    """filter_timeseries
    Filter timeseries based on
    1) whether the worm had skeletonised for more than 80%
    2) whether the length is between 0.5 and 2mm
    """

    def filter_worm(worm, frac_skel_thresh=0.8, length_int_um=[500, 2000]):
        """return True for good worms, False for bad worms"""
        frac_skel = (~worm['length'].isna()).sum() / worm['length'].shape[0]
        # if too many nans return bad straightaway
        if frac_skel < frac_skel_thresh:
            return False
        # otherwise check for average length
        median_length = np.nanmedian(worm['length'])
        if length_int_um[0] <= median_length <= length_int_um[1]:
            return True
        else:
            return False

    return ts.groupby('worm_index').filter(filter_worm).reset_index()


def make_well_id(df):
    assert all([c in df.columns for c in ID_COLS]), 'Missing some ID_COLS'
    well_id = df[ID_COLS].apply('_'.join, axis=1)
    return well_id


def intersect(long_list, short_list):
    return list(set(long_list).intersection(short_list))


def get_nonnumeric_cols(df):
    return df.select_dtypes(exclude=np.number).columns.to_list()


def get_nonnumorbool_cols(df):
    return df.select_dtypes(exclude=[bool, np.number]).columns.to_list()


def get_float64_cols(df):
    """This doesn't use select_dtypes as that may use more RAM as
    it returns a subset of the dataframe, not sure"""
    idx = df.dtypes == 'float64'
    return df.dtypes[idx].index

#%%
if __name__ == "__main__":
    RAW_DATA_DIR = Path('/Volumes/Ashur Pro2/DiseaseModel')
    METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/exploratory_metadata/EXPLORATORY_metadata_with_wells_annotations.csv')
    CONTROL_STRAIN = 'N2'
    CANDIDATE_GENE='cat-2'
    from helper import select_strains, make_colormaps, strain_gene_dict, DATES_TO_DROP
    from ts_helper import align_bluelight_meta
     
    timeseries_fname = RAW_DATA_DIR/ 'Results' / '{}_timeseries.hdf5'.format(CANDIDATE_GENE)
    is_reload_timeseries_from_results = True

    
    meta = pd.read_csv(METADATA_FILE, index_col=None)
    assert meta.worm_strain.unique().shape[0] == meta.worm_gene.unique().shape[0]
    meta.loc[:,'date_yyyymmdd'] = meta['date_yyyymmdd'].apply(lambda x: str(int(x)))
    #drop nan wells
    meta.dropna(axis=0,
                subset=['worm_gene'],
                inplace=True)
    # remove data from dates to exclude
    good_date = meta.query('@DATES_TO_DROP not in imaging_date_yyyymmdd').index
    # bad wells
    good_wells_from_gui = meta.query('is_bad_well == False').index
    meta = meta.loc[good_wells_from_gui & good_date,:]
    
    # only select strains of interest
    meta, idx, gene_list = select_strains(CANDIDATE_GENE,
                                          CONTROL_STRAIN,
                                          meta_df=meta,
                                          feat_df=None)
    strain_lut, stim_lut = make_colormaps(gene_list,
                                            idx,
                                            CANDIDATE_GENE,
                                            CONTROL_STRAIN,
                                            featlist=[])
    #strain to gene dictionary
    strain_dict = strain_gene_dict(meta)
    gene_dict = {v:k for k,v in strain_dict.items()}

    meta = align_bluelight_meta(meta)
    
    if is_reload_timeseries_from_results:
        # this uses tierpytools under the hood
        timeseries_df, hires_df  = load_bluelight_timeseries_from_results(
                            meta,
                            RAW_DATA_DIR / 'Results')
                            # save to disk
        timeseries_df.to_hdf(timeseries_fname, 'timeseries_df', format='table')
        hires_df.to_hdf(timeseries_fname, 'hires_df', format='table')
    else:  # from disk, then add columns
        # dataframe from the saved file
        timeseries_df = pd.read_hdf(timeseries_fname, 'timeseries_df')
        hires_df = pd.read_hdf(timeseries_fname, 'hires_df')
