#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 18:45:32 2021

Quick script to solve issues when downsampling timeseries data fails and files remain open

@author: tobrien
"""
import tables 
import pandas as pd
from pathlib import Path
flist = [f for f in Path.cwd().rglob('*hdf5')]
for fname in flist:
    with tables.File(fname, 'r+') as fid:
        pass
    foo = pd.read_hdf(fname, '/timeseries_df', mode='r+')
    print(foo.shape)
