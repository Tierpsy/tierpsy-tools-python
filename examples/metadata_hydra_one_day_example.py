#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:50:39 2019

@author: em812
"""

from tierpsytools.compile_metadata import merge_robot_metadata,get_day_metadata
from pathlib import Path

if __name__=="__main__":
    # Input
    day_root_dir = Path('/Volumes/behavgenom$/Ida/Data/Hydra/PilotDrugExps/AuxiliaryFiles/20191108_tierpsytools_dev')
    sourceplate_file = day_root_dir / '20191107_sourceplates.csv'
    manual_meta_file = day_root_dir / '20191108_manual_metadata.csv'
    
    # Save to
    #robot_metadata_file = day_root_dir.joinpath('20191107_robot_metadata.csv')
    metadata_file = day_root_dir.joinpath('20191108_day_metadata.csv')
    
    # Run
    robot_metadata = merge_robot_metadata(sourceplate_file, saveto=False)
    day_metadata = get_day_metadata(robot_metadata, manual_meta_file, saveto=metadata_file)
        

