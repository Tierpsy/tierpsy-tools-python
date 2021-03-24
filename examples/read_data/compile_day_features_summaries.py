#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compile tierpsy features and filenames summaries from different
days of experiment. It assumes that the individual summaries are inside the
day folders and that the day folders are named based on the date [yyyymmdd].

Created on Wed Jun 10 11:38:14 2020

@author: em812
"""

from pathlib import Path
from tierpsytools.read_data.compile_features_summaries import compile_tierpsy_summaries
import pdb

## Input

# - Results root
root = Path('/Volumes/behavgenom$/Tom/Data/Hydra/DiseaseModel/RawData/Results')
# - Paths to compiled files that will be created by this function
compiled_feat_file = root / 'features_summary_tierpsy_plate_filtered_traj_compiled.csv'
compiled_fname_file = root / 'filenames_summary_tierpsy_plate_filtered_traj_compiled.csv'

#%% Compile

# If the compiled files already exist, delete them
if compiled_feat_file.exists():
    compiled_feat_file.unlink()
if compiled_fname_file.exists():
    compiled_fname_file.unlink()

# Find all the day folders in Results
days = [item for item in root.glob('*/') if item.is_dir() and str(item.stem).startswith('20')]

# Find all the features and filenames summaries in the day folders
feat_files = [[file for file in day.glob('features*.csv')] for day in days]
fname_files = [[file for file in day.glob('filename*.csv')] for day in days]

# Make sure only one features summary file was detected in each day
assert all([len(flist)==1 for flist in feat_files])
# Make sure only one filenames summary file was detected in each day
assert all([len(flist)==1 for flist in fname_files])

feat_files = [flist[0] for flist in feat_files]
fname_files = [flist[0] for flist in fname_files]

# Compile the data from all the detected features and filenames summaries
compile_tierpsy_summaries(
    feat_files, compiled_feat_file, compiled_fname_file, fname_files=fname_files)