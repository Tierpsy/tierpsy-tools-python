#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to plot the full 96-well plate from a given RawVideo corresponding to a single camera view 
Provide a RawVideo filepath and a plot will be produced of the entire 96-well plate 
(imaged under 6 cameras simultaneously) for the first frame of the video

@author: sm5911
@date: 31/05/2023

"""

#%% Imports 

import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from tierpsytools.hydra.hydra_filenames_helper import CAM2CH_df, serial2channel, parse_camera_serial

#%% Globals

# Channel-to-plate mapping dictionary {'channel' : ((ax array location), rotate)}
CH2PLATE_dict = {'Ch1':((0,0),True),
                 'Ch2':((1,0),False),
                 'Ch3':((0,1),True),
                 'Ch4':((1,1),False),
                 'Ch5':((0,2),True),
                 'Ch6':((1,2),False)}

EXAMPLE_RAW_VIDEO_PATH = "/Volumes/hermes$/Saul/Keio_Screen/Data/Keio_Screen_Initial/RawVideos/20210406/keio_rep1_run1_bluelight_20210406_132006.22956809/000000.mp4"

FRAME = 0
DPI = 900

#%% Functions

def get_video_set(videofilepath):
    """ Get the set of filenames of the featuresN results files that belong to
        the same 96-well plate that was imaged under that rig """
            
    # get camera serial from filename
    camera_serial = parse_camera_serial(videofilepath)
    
    # get list of camera serials for that hydra rig
    hydra_rig = CAM2CH_df.loc[CAM2CH_df['camera_serial'] == camera_serial, 'rig']
    rig_df = CAM2CH_df[CAM2CH_df['rig'] == hydra_rig.values[0]]
    camera_serial_list = list(rig_df['camera_serial'])
   
    # extract filename stem 
    file_stem = str(videofilepath).split('.' + camera_serial)[0]
    name = videofilepath.name
    
    # get paths to RawVideo files
    file_dict = {}
    for camera_serial in camera_serial_list:
        channel = serial2channel(camera_serial)
        _loc, rotate = CH2PLATE_dict[channel]
        
        video_path = Path(file_stem + '.' + camera_serial) / name
        
        file_dict[channel] = video_path
        
    return file_dict
    
def plot_plate(videofilepath, save_path, frame=0, dpi=900): # frame = 'all'
    """ Tile first frame of raw videos for plate to create a single plot of the full 96-well plate, 
        correcting for camera orientation """

    file_dict = get_video_set(videofilepath)
    
    # define multi-panel figure
    columns = 3
    rows = 2
    h_in = 4
    x_off_abs = (3600-3036) / 3036 * h_in
    x = columns * h_in + x_off_abs
    y = rows * h_in
    
    x_offset = x_off_abs / x        # for bottom left image
    width = (1-x_offset) / columns  # for all but top left image
    width_tl = width + x_offset     # for top left image
    height = 1/rows                 # for all images
    
    plt.close('all')
    fig, axs = plt.subplots(rows, columns, figsize=[x,y])

    errlog = []
    print("Extracting frames...")
    for channel, video_path in tqdm(file_dict.items()):
        
        _loc, rotate = CH2PLATE_dict[channel]
        _ri, _ci = _loc

        # create bbox for image layout in figure
        if (_ri == 0) and (_ci == 0):
            # first image, bbox slightly shifted
            bbox = [0, height, width_tl, height]
        else:
            # other images
            bbox = [x_offset + width * _ci, height * (rows - (_ri + 1)), width, height]   
        
        # get location of subplot for camera
        ax = axs[_loc]
        
        vidcap = cv2.VideoCapture(str(video_path))
        success, img = vidcap.read()
        if success:
            if rotate:
                img = np.rot90(img, 2)   
                
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)        
        
        else:
            print("WARNING: Could not read video: '%s'" % (video_path))
            errlog.append(video_path)
        
        # set image position in figure
        ax.set_position(bbox)
        
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
                
    if len(errlog) > 0:
        print(errlog)
    
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path,
                    bbox_inches='tight',
                    pad_inches=0,
                    # transparent=True,
                    dpi=dpi)
            
    return

#%% Main
    
if __name__ == "__main__":
            
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--raw_video_path',
                        help="Path to a RawVideo file (single camera video) of plate to plot",
                        type=str, default=EXAMPLE_RAW_VIDEO_PATH)
    parser.add_argument('-s', '--save_dir', help="Path to save directory", type=str, default=None)
    args = parser.parse_args()
    
    RAW_VIDEO_PATH = Path(args.raw_video_path)
    SAVE_DIR = None if args.save_dir is None else Path(args.save_dir)
    
    print("Plotting plate for %s" % str(RAW_VIDEO_PATH))
    plot_plate(videofilepath=RAW_VIDEO_PATH, 
               save_path=(RAW_VIDEO_PATH.parent if SAVE_DIR is None else SAVE_DIR) /\
                   (RAW_VIDEO_PATH.stem + '.jpg'),
               frame=FRAME,
               dpi=DPI)
        
  