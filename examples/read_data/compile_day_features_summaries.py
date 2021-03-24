#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:38:14 2020

@author: em812
"""

from pathlib import Path
from tierpsytools.read_data.compile_features_summaries import compile_tierpsy_summaries
import pdb

root = Path('/Volumes/behavgenom$/Tom/Data/Hydra/DiseaseModel/RawData/Results')
compiled_feat_file = root / 'features_summary_tierpsy_plate_filtered_traj_compiled.csv'
compiled_fname_file = root / 'filenames_summary_tierpsy_plate_filtered_traj_compiled.csv'

if compiled_feat_file.exists():
    compiled_feat_file.unlink()
if compiled_fname_file.exists():
    compiled_fname_file.unlink()

days = [item for item in root.glob('*/') if item.is_dir() and str(item.stem).startswith('20')]

feat_files = [[file for file in day.glob('features*.csv')] for day in days]
fname_files = [[file for file in day.glob('filename*.csv')] for day in days]

assert all([len(flist)==1 for flist in feat_files])
assert all([len(flist)==1 for flist in fname_files])

feat_files = [flist[0] for flist in feat_files]
fname_files = [flist[0] for flist in fname_files]

compile_tierpsy_summaries(
    feat_files, compiled_feat_file, compiled_fname_file, fname_files=fname_files)