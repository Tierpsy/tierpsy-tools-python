#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:29:20 2019

@author: em812
"""
import re
import numpy as np
import pandas as pd

def column_from_well(wells_series):
    return wells_series.apply(lambda x: re.findall(r'\d+', x)[0]).astype(int)

def row_from_well(wells_series):
    return wells_series.apply(lambda x: re.findall(r'[^0-9]', x)[0])

def exract_randomized_by(robotlog,randomized_by):
    if 'column' in randomized_by:
        # extract the column number from the well number for mapping (as replicates are by column)
        robotlog[randomized_by] = column_from_well(robotlog['source_well'])
    elif 'row' in randomized_by:
        # extract the row number from the well number for mapping (if replicates are by row)
        robotlog[randomized_by] = row_from_well(robotlog['source_well'])
    elif ('well' in randomized_by) and (randomized_by != 'source_well'):
        robotlog[randomized_by] = robotlog['source_well']
    return robotlog

def rename_out_meta_cols(out_meta,randomized_by):

    out_meta = out_meta.rename(columns={'column':'source_plate_column',
                                            'row':'source_plate_row'})
    if 'column' in randomized_by:
        out_meta.rename(columns={randomized_by:'source_plate_column'})
    elif 'row' in randomized_by:
        out_meta.rename(columns={randomized_by:'source_plate_row'})
    elif ('well' in randomized_by) and (randomized_by != 'source_well'):
        out_meta = out_meta.drop(columns=[randomized_by])
    return out_meta

def explode_df(df, lst_col, fill_value='', preserve_index=False):

    # make sure `lst_cols` is list-like
    if (lst_col is not None
        and len(lst_col) > 0
        and not isinstance(lst_col, (list, tuple, np.ndarray, pd.Series))):
        lst_col = [lst_col]
    # all columns except `lst_col`
    idx_cols = df.columns.difference(lst_col)
    # calculate lengths of lists
    lens = df[lst_col[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_col}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res

def run_number_from_regex(file_list, run_number_regex=r'run\d+_'):
    """
    Get the run number from the name of the raw video based on a regex expression
    """
    run_name = [re.findall(run_number_regex, str(file.parent.parts[-1]))
                for file in file_list]
    run_name = [x[0] if len(x)==1 else '' for x in run_name ]
    imaging_run_number = [re.findall(r'\d+', x) for x in run_name]
    imaging_run_number = [int(x[0]) if len(x)==1 else np.nan for x in imaging_run_number]
    if np.any(np.isnan(imaging_run_number)):
        raise Exception('Run number could not be recovered from all raw video imgstore names.')
    return imaging_run_number

def run_number_from_timestamp(file_list, camera_serial):
    """
    Get the run number based on the camera serial and the timestamp.
    !! This function is not safe, so the option to get the run number based
    in the timestamp is not integrated to the standard functions to
    compile metadata for hydra
    """
    timestamp = [int(file.parent.stem.split('_')[-1]) for file in file_list]
    bluelight = []
    for file in file_list:
        blue = [b for b in ['bluelight', 'prestim', 'poststim'] if b in file.parent.stem]
        if len(blue)==1:
            blue = blue[0]
        elif len(blue)==0:
            blue = 'nobluelight'
        elif len(blue)>1:
            raise Exception('More than one bluelight label in video {}.'.format(file))
        bluelight.append(blue)

    df = pd.DataFrame({
        'bluelight': bluelight,
        'file_name':file_list,
        'camera_serial': camera_serial,
        'timestamp': timestamp
        })

    df = df.groupby(['camera_serial', 'bluelight']).apply(
        lambda x: x.sort_values(by=['timestamp'], ascending=True).assign(
            imaging_run_number=list(range(1,x.shape[0]+1))
            ))

    df = df.reset_index(level=2).sort_values(by='level_2')
    assert all([x==y for x,y in zip(file_list, df['file_name'].to_list())])

    return df['imaging_run_number'].to_list()


if __name__=="__main__":
    df = pd.DataFrame([[0,1],[1,2]],columns=['a','b'])
    df['c'] = pd.Series([[0],[0,1]])
    df['d'] = pd.Series([[],[0,1]])
    df1 = explode_df(df,['c'])
    print(df1)
    df2 = explode_df(df,['d'])
    print(df2)