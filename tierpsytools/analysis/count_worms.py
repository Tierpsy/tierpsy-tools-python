#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 19:50:33 2021

@author: lferiani
"""

import pandas as pd

from tierpsytools.read_data.get_timeseries import read_timeseries


def n_worms_per_frame(timestamp, max_t: int = None) -> pd.Series:
    """
    n_worms_per_frame Take an array or pd.Series with timestamps, as you'd get
    from Tierpsy, and determine how many worms were detected over time.
    The rationale behind the maths is that if a timestamp appears twice, two
    worms were detected at the same time, so n=2 for that frame.
    This function also makes sure that if no worms were detected in frame i,
    this is reflected in the results.

    Parameters
    ----------
    timestamp : iterable
        frame numbers from the timeseries or trajectories in tierpsy's results
    max_t : int, optional
        number of frames in a video, by default None
        If empty, the maximum value of timestamp will be used.
        It is important to use `max_t` to account for the edge case in which no
        worms are detected at the end of the video

    Returns
    -------
    pd.Series
        number of worms tracked (values) versus time (series index).
        The output of this function can then be passed on to
        `fraction_of_time_with_n_worms`
    """

    # use pandas series. could use numpy but only > 1.18
    if not isinstance(timestamp, pd.Series):
        import pdb
        pdb.set_trace()
        timestamp = pd.Series(timestamp)

    # if timestamp is all nan, then that's likely a well that was never recorded
    if timestamp.isna().all():
        # that won't affect the function actually,
        # just that I cannot do the next check
        pass
    else:
        # check all integers - easier than testing all type of int dtypes
        # assert timestamp.astype(str).str.isdigit().all(), (
        #     "Need to use integer timestamps")
        if any(timestamp % 1 != 0):
            err_msg = 'Non integer timestamp found:\n'
            err_msg += f'{timestamp[timestamp % 1 != 0]}'
            err_msg += '\nANeed to use integer timestamps'
            raise ValueError(err_msg)


    if max_t is None:
        max_t = timestamp.max()
    if max_t != max_t:
        # max_t is nan
        max_t = 0

    # count how many repeated values for each entry in timestamp
    n_worms_vs_time = timestamp.value_counts().sort_index()

    # the above will not count zeros for missing values
    n_worms_vs_time = n_worms_vs_time.reindex(
        pd.Index(range(max_t + 1), name='timestamp'),
        fill_value=0
        )
    n_worms_vs_time.name = 'n_worms'

    return n_worms_vs_time


def _fraction_of_time_with_n_worms(
        n_worms_vs_time: pd.Series, max_n: int = None) -> pd.Series:
    """
    fraction_of_time_with_n_worms Given a pd.Series with
        number of worms tracked (values) versus time (series index),
        calculate the fraction of time that 0, 1, .., n worms were detected


    Parameters
    ----------
    n_worms_vs_time : pd.Series
        number of worms tracked (values) versus time (series index)
        output of `n_worms_per_frame`
    max_n : int, optional
        maximum number of worms expected in a frame, by default None
        If empty, the maximum number of worms detected will be used.
        It is important to set max_n to account for the case in which tierpsy
        never detects all the worms in a well at the same time.

    Returns
    -------
    pd.Series
        what fraction of time/a video (values) tierpsy tracked n worms (index)
    """

    if max_n is None:
        max_n = n_worms_vs_time.max()

    # what proportion of a video did we see 0, 1, 2 worms?
    # sort=None sorts by index, which is the number of worms
    # normalize=True divides by the total number of values
    time_fraction_n_worms = n_worms_vs_time.value_counts(
        sort=False, normalize=True)

    # there might have been gaps in the numer of worms before
    # (e.g. only ever 1 or 3 worms seen at same time)
    # which is ok for the normalisation but bad for plotting later
    time_fraction_n_worms = time_fraction_n_worms.reindex(
        pd.Index(range(max_n + 1), name='n_worms'),
        fill_value=0
        )
    time_fraction_n_worms.name = 'time_fraction'

    return time_fraction_n_worms


def get_frequency_of_detecting_n_worms_in_one_well(
        timestamp, max_t: int = None, max_n: int = None) -> pd.Series:
    """
    get_frequency_of_detecting_n_worms_in_one_well Take an array or pd.Series
        with timestamps, as you'd get from Tierpsy, and calculate what fraction
        of the time 0, 1, 2, ... max_n worms were detected at the same time.

    This is mostly a wrapper for the functions `n_worms_per_frame`

    Parameters
    ----------
    timestamp : iterable
        frame numbers from the timeseries or trajectories in tierpsy's results
    max_t : int, optional
        number of frames in a video, by default None
        If empty, the maximum value of timestamp will be used.
        It is important to use `max_t` to account for the edge case in which no
        worms are detected at the end of the video
    max_n : int, optional
        maximum number of worms expected in a frame, by default None
        If empty, the maximum number of worms detected will be used.
        It is important to set max_n to account for the case in which tierpsy
        never detects all the worms in a well at the same time.

    Returns
    -------
    pd.Series
        what fraction of a video (values) tierpsy tracked n worms (index)

    """
    nw = n_worms_per_frame(timestamp, max_t=max_t)
    ft = _fraction_of_time_with_n_worms(nw, max_n=max_n)
    return ft


def get_frequency_of_detecting_n_worms(
        onevideo_meta: pd.DataFrame, max_t: int = None, max_n: int = None
        ) -> pd.Series:
    """
    get_frequency_of_detecting_n_worms Given a pandas dataframe containing
        the metadata relative to a single video, read the timeseries_data
        from the files in the `feats_fname` column, and on a well-by-well basis
        calculate what fraction of time tierpsy detected 0, 1, ..., `max_n`
        worms at the same time.
        Will not work with featuresN files generated with Tierpsy < 1.5.2

    Parameters
    ----------
    onevideo_meta : pd.DataFrame
        metadata pertaining to one video only. Must contain a column called
        `feats_fname` with the full path to the features file
    max_t : int, optional
        number of frames in a video, by default None
        If empty, the maximum value of timestamp will be used.
        It is important to use `max_t` to account for the edge case in which no
        worms are detected at the end of the video
    max_n : int, optional
        maximum number of worms expected in a frame, by default None
        If empty, the maximum number of worms detected will be used.
        It is important to set max_n to account for the case in which tierpsy
        never detects all the worms in a well at the same time.

    Returns
    -------
    pd.Series
        for each well in the video's timeseries,
        what fraction of the video (values) tierpsy tracked n worms (index)
    """

    # check the full path to the timeseries exists
    assert 'feats_fname' in onevideo_meta, (
        '`onevideo_meta` must contain a column '
        'with the full path to the features file'
        )
    # check it's a chunk of metadata that only contains data from the one video
    assert onevideo_meta['feats_fname'].nunique() == 1, (
        '`onevideo_meta` cannot contain metadata from multiple videos')
    fname = onevideo_meta['feats_fname'].values[0]

    # wells to read
    if 'well_name' in onevideo_meta:
        only_wells = onevideo_meta['well_name'].to_list()
    else:
        only_wells = None

    # load timeseries
    ts = read_timeseries(
        fname, names=['well_name', 'timestamp'], only_wells=only_wells)

    # if no worms were ever detected in a well, they'll be missing from here.
    # so add them back in, so that the other functions can correctly return
    # that 0 worms were seen throughout the whole video
    ts = pd.merge(
        ts, onevideo_meta[['well_name']], on='well_name', how='right')

    # count fraction of time per each well
    ft = ts.groupby('well_name')['timestamp'].apply(
        get_frequency_of_detecting_n_worms_in_one_well,
        max_t=max_t,
        max_n=max_n
        ).rename('time_fraction')

    expected_n_rows = onevideo_meta.shape[0] * (max_n+1)
    actual_n_rows = ft.shape[0]
    assert expected_n_rows == actual_n_rows, (
        f'Output has {actual_n_rows}, was expecting {expected_n_rows}.')

    return ft


def _test_toy():
    # toy data
    max_t = 7
    max_n = 4
    x = [1, 1, 1, 3, 3, 4, 4, 6, 6, 6]
    y = n_worms_per_frame(x, max_t=max_t)
    z = _fraction_of_time_with_n_worms(y, max_n=max_n)

    zz = get_frequency_of_detecting_n_worms_in_one_well(
        x, max_t=max_t, max_n=max_n)

    expected = pd.Series([0.5, 0, 0.25, 0.25, 0], name='time_fraction')
    expected.index.name = 'n_worms'
    assert z.equals(expected)
    assert z.equals(zz)


if __name__ == '__main__':
    _test_toy()
