#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:41:09 2020

@author: lferiani
"""

import numpy as np
from warnings import warn
from pathlib import Path
from tierpsytools.hydra.hydra_filenames_helper import find_imgstore_videos
import re
import pandas as pd


def get_imgstore_root(imgstore_name):
    if 'bluelight' in imgstore_name:
        token = 'bluelight'
    elif 'prestim' in imgstore_name:
        token = 'prestim'
    elif 'poststim' in imgstore_name:
        token = 'poststim'
    else:
        warn(f'No match for {imgstore_name}')
        token = ''
        return ''
    splitted = imgstore_name.split(token)
    # if no token, no split, hence return nothing
    if len(splitted) == 1:
        warn(f'odd')
    # else return everything before
    imgstore_root = splitted[0].strip('_')
    return imgstore_root


def parse_imaging_datetime(filename):
    regex = r"20\d{6}\_\d{6}(?=\.\d{8})"
    datetime = re.findall(regex, str(filename).lower())[0]
    return datetime


def hhmmss2s(hhmmss):
    assert len(hhmmss) == 6
    hh = int(hhmmss[:2])
    mm = int(hhmmss[2:4])
    ss = int(hhmmss[4:])
    s = 3600*hh + 60*mm + ss
    return s


def check_time_differences(df, tol=3):
    diff1 = (df['time_bluelight'].apply(hhmmss2s)
             - df['time_prestim'].apply(hhmmss2s))
    diff2 = (df['time_poststim'].apply(hhmmss2s)
             - df['time_bluelight'].apply(hhmmss2s))

    imgstore_cols = [c for c in df.columns if c.startswith('imgstore')]
    ind_dodgy = []
    if diff1.max() > diff1.mean() + tol/2:
        ind_dodgy.extend(
            (diff1 > diff1.mean() + tol/2).to_numpy().nonzero()[0]) #nonzeros
    if diff1.min() < diff1.mean() - tol/2:
        ind_dodgy.extend(
            (diff1 < diff1.mean() - tol/2).to_numpy().nonzero()[0])
    if diff2.max() > diff2.mean() + tol/2:
        ind_dodgy.extend(
            (diff2 > diff2.mean() + tol/2).to_numpy().nonzero()[0])
    if diff2.min() < diff2.mean() - tol/2:
        ind_dodgy.extend(
            (diff2 > diff2.mean() - tol/2).to_numpy().nonzero()[0])
    dodgy = df.loc[diff2.idxmin(), imgstore_cols]
    print('Double-check matching of {}'.format(dodgy))


def get_triplet(df_g, parentpath=False):

    assert df_g.shape[0] % 3 == 0, \
        f'{df_g.shape[0]} videos matching{df_g.name}, not divisible by 3'

    if df_g.shape[0] != 3:
        warn(f'Found > 3 videos matching {df_g.name}.'
             ' Matching based on timestamp.')
    sorted_df_g = df_g.sort_values('time', ascending=True)
    out_df = pd.DataFrame()
    out_df['imgstore_prestim'] = sorted_df_g['imgstore'].values[0::3]
    out_df['time_prestim'] = sorted_df_g['time'].values[0::3]
    out_df['imgstore_bluelight'] = sorted_df_g['imgstore'].values[1::3]
    out_df['time_bluelight'] = sorted_df_g['time'].values[1::3]
    out_df['imgstore_poststim'] = sorted_df_g['imgstore'].values[2::3]
    out_df['time_poststim'] = sorted_df_g['time'].values[2::3]
    # print(len(sorted_df_g['parent_path'].unique()))
    if parentpath:
        out_df.loc[:, 'parent_path'] = sorted_df_g['parent_path'].unique()

    return out_df


def match_bluelight_videos(df, fullpath=False):
    """
    match_bluelight_videos: return the matching prestim/bluelight/poststim
    triplets of videos typical of a bluelight-screening

    Parameters
    ----------
    df : pandas.DataFrame
        The output of find_imgstore_videos, from hydra_filenames_helper.py

    Returns
    -------
    match_df : pandas.DataFrame
        DataFrame containing, for each triplet of imgstore files:
            1)  imgstore_root (the name typed in when imaging)
            2)  rig (Hydra01 to Hydra05)
            3)  camera_serial (unique id of camera)
            4)  channel (Ch1 to Ch6, position of camera in the rig)
            5)  date_YYYYMMDD (imaging day)
            6)  imgstore_prestim (the name of the folder containing
                metadata.yaml, for the pre-stimulus video)
            7)  time_prestim (imaging time for pre-stimulus video, HHMMSS)
            8)  imgstore_bluelight
            9)  time_bluelight
            10) imgstore_poststim
            11) time_poststim
            12) parent_path (full path to the parent of the imgstores,
                ususally will be the day folder)
    """
    # isolating part of filename before prestim/bluelight/poststim
    df['imgstore_root'] = df['imgstore'].apply(get_imgstore_root)
    # get day and time
    df['datetime'] = df['imgstore'].apply(parse_imaging_datetime)
    df[['date_YYYYMMDD', 'time']] = df['datetime'].str.split(pat='_', n=1,
                                                             expand=True)
    df.drop(columns=['datetime'], inplace=True)

    # get the path (not down to the file). This way it will be conserved after
    # the next steps
    if fullpath:
        df['parent_path'] = df['full_path'].apply(lambda x: x.parent.parent)

    # group df by variables that *should* isolate a video triplet:
    # imgstore_root (inputted during experiment), rig, camera_serial, date
    # if >3 videos per group, matching is done by timestamp
    cols_to_grpby = ['imgstore_root', 'rig', 'camera_serial', 'channel',
                     'date_YYYYMMDD']
    match_df = df.groupby(cols_to_grpby).apply(get_triplet)
    # clean up the returned dataframe
    match_df.reset_index(inplace=True)
    # there' spurious "level_n" columns now
    col_spurious = [c for c in match_df.columns if c.startswith('level_')]
    match_df.drop(columns=col_spurious, inplace=True)
    # there shouldn't be any, but just in case:
    match_df.drop_duplicates(inplace=True)

    # checks:
    check_time_differences(match_df, tol=3)
    return match_df


def match_bluelight_videos_in_folder(target_dir):
    """
    match_bluelight_videos_in_folder: scan target_dir, return matching
    prestim/bluelight/poststim triplets of videos.

    Parameters
    ----------
    target_dir : pathlib.Path, or str
        A folder with imgstores (in it or in its subfolders)

    Returns
    -------
    match_df : pandas.DataFrame
        DataFrame containing, for each triplet of imgstore files:
            1)  imgstore_root (the name typed in when imaging)
            2)  rig (Hydra01 to Hydra05)
            3)  camera_serial (unique id of camera)
            4)  channel (Ch1 to Ch6, position of camera in the rig)
            5)  date_YYYYMMDD (imaging day)
            6)  imgstore_prestim (the name of the folder containing
                metadata.yaml, for the pre-stimulus video)
            7)  time_prestim (imaging time for pre-stimulus video, HHMMSS)
            8)  imgstore_bluelight
            9)  time_bluelight
            10) imgstore_poststim
            11) time_poststim
            12) parent_path (full path to the parent of the imgstores,
                ususally will be the day folder)
    """
    df = find_imgstore_videos(target_dir)
    matched_bluelight_videos_df = match_bluelight_videos(df)
    return matched_bluelight_videos_df


# %%
if __name__ == '__main__':

    wd = Path('/Volumes/behavgenom$/Ida/Data/Hydra'
              + '/SyngentaStrainScreen/RawVideos')

    wd = Path('/Volumes/behavgenom$/Ida/Data/Hydra/DiseaseScreen/RawVideos')
    # call function that already finds all imgstores and put them in neat df
    df = match_bluelight_videos_in_folder(wd)
