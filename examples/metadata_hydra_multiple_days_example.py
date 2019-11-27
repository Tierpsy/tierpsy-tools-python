#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive compiling of metadata from many days of experiments

Created on Fri Nov 15 15:50:39 2019

@author: em812
"""

from tierpsytools.compile_metadata import merge_robot_metadata,get_day_metadata,concatenate_days_metadata
from pathlib import Path

if __name__=="__main__":
    # Input
    aux_root_dir = Path('/Volumes/behavgenom$/Ida/Data/Hydra/PilotDrugExps/AuxiliaryFiles')
    day_root_dirs = [d for d in aux_root_dir.glob("*") if d.is_dir()]
    
    sourceplate_files = [[file for file in d.glob('*_sourceplates.csv')] for d in day_root_dirs]
    manual_meta_files = [[file for file in d.glob('*_manual_metadata.csv')] for d in day_root_dirs]
    
    # Saveto
    metadata_files = [d.joinpath('{}_day_metadata.csv'.format(d.stem)) for d in day_root_dirs]
    
    # Run compilation of day metadata for all the days
    for day,source,manual_meta,saveto in zip(day_root_dirs,sourceplate_files,manual_meta_files,metadata_files):
        if len(source)!=1:
            print('There is not a unique sourceplates file in day {}. Metadata cannot be compiled'.format(day))
            continue
        if len(manual_meta)!=1:
            print('There is not a unique manual_metadata file in day {}. Metadata cannot be compiled'.format(day))
            continue
        robot_metadata = merge_robot_metadata(source[0], saveto=False)
        day_metadata = get_day_metadata(robot_metadata, manual_meta[0], saveto=saveto[0])
    
    # Conactenate all metadata
    concatenate_days_metadata(aux_root_dir)