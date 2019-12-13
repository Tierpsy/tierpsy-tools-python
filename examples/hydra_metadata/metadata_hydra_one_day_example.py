#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:50:39 2019

@author: em812
"""

from tierpsytools.hydra.compile_metadata import merge_robot_metadata
from tierpsytools.hydra.compile_metadata import get_day_metadata
from tierpsytools import EXAMPLES_DIR
from pathlib import Path

if __name__=="__main__":
    # Input
    data_dir = Path(EXAMPLES_DIR) / 'hydra_metadata' / 'data'
    day_root_dir = data_dir / 'AuxiliaryFiles' / 'day1'
    sourceplate_file = day_root_dir / '20191107_sourceplates.csv'
    manual_meta_file = day_root_dir / '20191108_manual_metadata.csv'
    
    # Save to
    #robot_metadata_file = day_root_dir.joinpath('20191107_robot_metadata.csv')
    metadata_file = day_root_dir / '{}_day_metadata.csv'.format(
        day_root_dir.stem)
    
    # Run
    robot_metadata = merge_robot_metadata(sourceplate_file, saveto=False)
    day_metadata = get_day_metadata(
        robot_metadata, manual_meta_file, saveto=metadata_file,
        del_if_exists=True, include_imgstore_name = True, 
        raw_day_dir=data_dir / 'RawVideos' / 'day1')
        

