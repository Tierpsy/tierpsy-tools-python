#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:29:29 2020

@author: ibarlow

Functions for exporting list of masked videos.

Also checks that all the videos have been processed

Generates a file called missing videos if there are missing files 

"""

from pathlib import Path
import pandas as pd
import re
import numpy as np
import warnings
from tierpsytools.hydra.compile_metadata import add_imgstore_name
from tierpsytools.hydra import CAM2CH_df

N_CAMERAS = 6
N_RIGS = 5

total_cameras = N_CAMERAS * N_RIGS
run_regex = r"(?<=run)\d{1,2}"
OVERWRITE = False
imgstore_regex = r"\d{8,}(?=/metadata)"
date_regex = r"\d{8}"

#%%
def get_prestim_videos(day_root_dir):
    """
    Get list of prestim videos
    
    Parameters
    ----------
    day_root_dir : Path to root directory of masked videos for a day of tracking
        DESCRIPTION.

    Returns
    -------
    prestim_videos : dataframe
        DESCRIPTION.

    """
    masked_videos = list(day_root_dir.rglob('metadata.hdf5'))
    prestim_videos = [f for f in masked_videos if
                      'prestim' in str(f)]

    # put the prestim videos in a df for exporting
    prestim_videos = pd.DataFrame({'masked_video': prestim_videos})
    prestim_videos['run_number'] = [int(re.search(run_regex,
                                    str(r.masked_video))[0])
                                    for i, r in
                                    prestim_videos.iterrows()]
    prestim_videos['imgstore_camera'] = [
                                str(re.search(imgstore_regex,
                                              str(r.masked_video))
                                                [0]) for i, r in
                                        prestim_videos.iterrows()
                                        ]
    prestim_videos = pd.merge(prestim_videos,
                             CAM2CH_df,
                             left_on='imgstore_camera',
                             right_on='camera_serial',
                             )
    prestim_videos.drop(columns='imgstore_camera',
                        inplace=True)
    prestim_videos.sort_values(by=['run_number', 'rig', 'channel'], inplace=True)

    return prestim_videos
# %%
def missing_videos(prestim_videos, day_root_dir, CAMERAS_PER_RIG=6, overwrite=False):
    """
    Checks that all cameras for each run have been processed
    
    Parameters
    ----------
    prestim_videos : TYPE
        DESCRIPTION.
    day_root_dir : TYPE
        DESCRIPTION.
    CAMERAS_PER_RIG : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    missing_videos : TYPE
        DESCRIPTION.

    """
        
    outfile = day_root_dir.parent.parent / 'AuxiliaryFiles' / \
        '{}_missing_videos.csv'.format(day_root_dir.stem)
    
    if outfile.exists():
        if overwrite:
            warnings.warn('{} already exists and will be overwritten'.format(outfile))
            outfile.unlink()
        else:
            warnings.warn('{} exists and will not be saved'.format(outfile))
      

    missing_videos = []
    prestim_videos_grouped = prestim_videos.groupby('run_number')
    
    _resid = (prestim_videos_grouped.apply(len) % CAMERAS_PER_RIG > 0).reset_index()
    _resid.rename(columns={0: 'cameras_missing'}, inplace=True)
    
    runs_to_check = list(_resid[_resid.cameras_missing == True]['run_number'])
    for r in runs_to_check:
        
        _checking = prestim_videos_grouped.get_group(r)
        rigs_running = prestim_videos_grouped.get_group(r)['rig'].unique()       
        cameras = CAM2CH_df.query('@rigs_running in rig')['camera_serial']
        
        missing_cameras = set(cameras) - set(_checking['camera_serial'])
        missing_videos.append((r,
                               [str(m) for m in
                                missing_cameras]))

    missing_videos = pd.DataFrame(missing_videos,
                                  columns=['run_number',
                                           'camera'])

    if missing_videos.empty:
        return
    
    else:
        
        if not outfile.exists() and overwrite == False:
            missing_videos.to_csv(outfile,
                                  index=False)  
        
        return missing_videos
   
    
#%%

def check_all_videos(prestim_videos, day_root_dir, N_CAMERAS=N_CAMERAS):
    """
    check that all the cameras were recording
    
    Parameters
    ----------
    prestim_vidoes : TYPE
        dataframe of prestim videos.
    
    day_root_dir : TYPE
        DESCRIPTION.
    N_CAMERAS : TYPE, optional
        DESCRIPTION. The default is total_cameras.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
        
    imgstore_cameras = list(prestim_videos['camera_serial'].unique())        
    
    day = day_root_dir.stem

    if len(imgstore_cameras) < (N_CAMERAS):
        N_CAMERAS = len(imgstore_cameras)
        
        not_recording = set(CAM2CH_df.camera_serial) - set(imgstore_cameras)
        
        print('Not all cameras were recording on this day: {}'.format(not_recording))
        
        return not_recording
    
    elif prestim_videos.shape[0] % N_CAMERAS != 0:

        print('Not all rigs running, checking missing videos for {}'.format(day))
        
        return missing_videos(prestim_videos, day_root_dir)
   
    else:
        return None
        
# %% example uses   

if __name__ == '__main__':
    INPUT_DIR = Path('/Volumes/behavgenom$/Ida/Data/Hydra/DiseaseScreen/MaskedVideos')
    
    day_dirs = [d for d in INPUT_DIR.glob('*') if d.is_dir()
                and re.search(date_regex, str(d)) is not None]
    
    for d in day_dirs:
        prestim = get_prestim_videos(d)
        check_all_videos(prestim, d)
        
        prestim.to_csv(d.parent.parent / 'AuxiliaryFiles' / d.stem / \
                       '{}_masked_videos_to_check.csv'.format(d.stem),
                       index=False)
