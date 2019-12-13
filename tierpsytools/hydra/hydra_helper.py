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
    
    # make sure `lst_cols` is list-alike
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

if __name__=="__main__":
    df = pd.DataFrame([[0,1],[1,2]],columns=['a','b'])
    df['c'] = pd.Series([[0],[0,1]])
    df['d'] = pd.Series([[],[0,1]])
    df1 = explode_df(df,['c'])
    print(df1)
    df2 = explode_df(df,['d'])
    print(df2)