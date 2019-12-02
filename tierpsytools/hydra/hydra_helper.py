#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:29:20 2019

@author: em812
"""
import re

def column_from_well(wells_series):
    return wells_series.apply(lambda x: re.findall(r'\d+', x)[0])

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
    
    