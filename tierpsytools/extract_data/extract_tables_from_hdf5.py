#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:30:22 2020

@author: lferiani
"""

import h5py
import tables
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path


def ask_user_go_ahead(fnames):
    """ask_user_go_ahead
    trivial UI loop asking to go ahead, stop, or print files
    """
    print('Found {} files'.format(len(fnames)))
    # loop for user interaction
    user_input = 'a'
    while user_input not in ['y', 'n']:
        user_input = input("Continue? [Y/n, p prints the files list] ").lower()
        if user_input == 'n':
            print('Execution stopped')
            return False
        elif user_input == 'y':
            return True
        elif user_input == 'p':
            for fname in fnames:
                print(fname)


def copy_table(table_name, src_fname, dst_fname):
    """copy_table
    Copy a single hdf5 table (table_name is the path within the hdf5 file)
    from src_fname to dst_fname
    """
    if not dst_fname.parent.exists():
        dst_fname.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_fname, 'r') as fids:
        with h5py.File(dst_fname, 'w') as fidd:
            fids.copy(table_name, fidd['/'])


def extract_table(table_name, src, dst, search_string):
    """extract_table
    find files in dst according to search_string,
    ask user for go ahead,
    loop over files - create destination folder if needed and copy table
    """
    # find files
    fnames = list(src.rglob(search_string))
    # user input
    is_ok = ask_user_go_ahead(fnames)
    # exit early if user aborted
    if not is_ok:
        print('Nothing done')
        return
    # main loop
    for fname in tqdm(fnames):
        dst_fname = dst / fname.relative_to(src)
        dst_fname.parent.mkdir(parents=True, exist_ok=True)
        copy_table(table_name, fname, dst_fname)


if __name__ == '__main__':

    # src = Path('/Volumes/behavgenom$/Priota/Data/PaschalisChicagoTracking')
    # dst = Path('/Users/lferiani/work_repos/tierpsytools_python/data/featuresN_data_extraction')
    # table_name = '/trajectories_data'
    # search_string = '*_featuresN.hdf5'

    src = Path('/Users/lferiani/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY_forGUI/MaskedVideos')
    dst = Path('/Users/lferiani/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY_forGUI/wells_annotations')
    table_name = '/fov_wells'
    search_string = '*.hdf5'
    # search_string = '*prestim*/*.hdf5'

    extract_table(table_name, src, dst, search_string)
