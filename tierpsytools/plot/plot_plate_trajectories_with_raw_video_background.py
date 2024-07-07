#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot 96-well Plate Trajectories

A script to plot trajectories for worms tracked in all the wells of the 96-well
plate to which that video belongs. Just provide a featuresN filepath from
Tierpsy filenames summaries and a plot will be produced of tracked worm
trajectories throughout the video, for the entire 96-well plate (imaged under
6 cameras simultaneously).

@author: sm5911 (Saul Moore)
@date: 23/06/2020

"""

#%% Imports
import cv2
import sys
import h5py
import argparse
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tierpsytools.hydra.hydra_filenames_helper import CAM2CH_df, serial2channel, parse_camera_serial

#%% Channel-to-plate mapping dictionary (global)

# {'channel' : ((ax array location), rotate)}
CH2PLATE_dict = {'Ch1':((0,0),True),
                 'Ch2':((1,0),False),
                 'Ch3':((0,1),True),
                 'Ch4':((1,1),False),
                 'Ch5':((0,2),True),
                 'Ch6':((1,2),False)}


#%% Functions

def get_trajectory_data(featuresfilepath):
    """ Read Tierpsy-generated featuresN file trajectories data and return
        the following info as a dataframe:
        ['x', 'y', 'frame_number', 'worm_id'] """

    # Read HDF5 file + extract info
    with h5py.File(featuresfilepath, 'r') as f:
        df = pd.DataFrame({'x': f['trajectories_data']['coord_x'],\
                           'y': f['trajectories_data']['coord_y'],\
                           'frame_number': f['trajectories_data']['frame_number'],\
                           'worm_id': f['trajectories_data']['worm_index_joined']})
    return(df)


def plot_trajectory(featurefilepath,
                   ax=None,
                   downsample=10,
                   legend=True,
                   rotate=False,
                   img_shape=None,
                   **kwargs):
    """ Overlay feature file trajectory data onto existing figure """

    df = get_trajectory_data(featurefilepath)

    if not ax:
        fig, ax = plt.subplots(**kwargs)

    # Rotate trajectories when plotting a rotated image
    if rotate:
        if not img_shape:
            raise ValueError('Image shape missing for rotation.')
        else:
            height, width = img_shape[0], img_shape[1]
            df['x'] = width - df['x']
            df['y'] = height - df['y']

    # Downsample frames for plotting
    if downsample <= 1 or downsample == None: # input error handling
        ax.scatter(x=df['x'], y=df['y'], s=10, c=df['frame_number'],\
                   cmap='plasma')
    else:
        ax.scatter(x=df['x'][::downsample], y=df['y'][::downsample],\
                   s=10, c=df['frame_number'][::downsample], cmap='plasma')
    #ax.tick_params(labelsize=5)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if legend:
        _legend = plt.colorbar(pad=0.01)
        _legend.ax.get_yaxis().labelpad = 10 # legend spacing
        _legend.ax.set_ylabel('Frame Number', rotation=270, size=7) # legend label
        _legend.ax.tick_params(labelsize=5)

    ax.autoscale(enable=True, axis='x', tight=True) # re-scaling axes
    ax.autoscale(enable=True, axis='y', tight=True)


def get_video_set(featurefilepath):
    """ Get the set of filenames of the featuresN results files that belong to
        the same 96-well plate that was imaged under that rig """

    # get camera serial from filename
    camera_serial = parse_camera_serial(featurefilepath)

    # get list of camera serials for that hydra rig
    hydra_rig = CAM2CH_df.loc[CAM2CH_df['camera_serial']==camera_serial,'rig']
    rig_df = CAM2CH_df[CAM2CH_df['rig']==hydra_rig.values[0]]
    camera_serial_list = list(rig_df['camera_serial'])

    # extract filename stem
    file_stem = str(featurefilepath).split('.' + camera_serial)[0]

    file_dict = {}
    for camera_serial in camera_serial_list:
        channel = serial2channel(camera_serial)

        ch_featfilepath = Path(file_stem + '.' + camera_serial)
        ch_featfilepath /= 'metadata_featuresN.hdf5'

        file_dict[channel] = ch_featfilepath

    return file_dict


def feat2raw(featfilepath):
    """Get imgstore name that corresponds to a results video"""
    rawfilepath = Path(
        str(featfilepath.parent).replace('Results', 'RawVideos')
        ) / 'metadata.yaml'
    assert rawfilepath.exists(), f'Cannot find imgstore for {featfilepath}'
    return rawfilepath


def get_frame_from_raw(rawvidname):
    
    from tierpsy.analysis.compress.selectVideoReader import selectVideoReader

    vid = selectVideoReader(str(rawvidname))
    status, frame = vid.read_frame(0)
    assert status == 1, f'Something went wrong while reading {rawvidname}'
    return frame

def plot_6wellplate(img,is_rotate180=False, ax=None, line_thickness=20):
  
        """
        Plot the fitted wells, the wells separation, and the name of the well.
        (only if these things are present!)"""

        # make sure I'm not working on the original image
        if is_rotate180:
            # a rotation is 2 reflections
            _img = cv2.cvtColor(img.copy()[::-1, ::-1],
                                cv2.COLOR_GRAY2BGR)
        else:
            _img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
#        pdb.set_trace()
        # flags: according to dataframe state, do or do not do
        # add names of wells
        # plot, don't close
        if not ax:
            figsize = (8, 8*_img.shape[0]/_img.shape[1])
            fig = plt.figure(figsize=figsize)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
        else:
            fig = ax.figure
            ax.set_axis_off()

        ax.imshow(_img)
     
#        plt.axis('off')
        # plt.tight_layout()
        return fig
def plot_plate_trajectories(featurefilepath, saveDir=None, downsample=10, fov= False):
    """ Tile plots and merge into a single plot for the
        entire 96-well plate, correcting for camera orientation. """

    from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter

    file_dict = get_video_set(featurefilepath)

    # define multi-panel figure
    columns = 3
    rows = 2
    h_in = 6
    x_off_abs = (3600-3036) / 3036 * h_in
    x = columns * h_in + x_off_abs
    y = rows * h_in
    fig, axs = plt.subplots(rows,columns,figsize=[x,y])

    x_offset = x_off_abs / x  # for bottom left image
    width = (1-x_offset) / columns  # for all but top left image
    width_tl = width + x_offset   # for top left image
    height = 1/rows        # for all images

    for channel, ch_featfilepath in file_dict.items():

        _loc, rotate = CH2PLATE_dict[channel]
        _ri, _ci = _loc

        # create bbox for image layout in figure
        if (_ri == 0) and (_ci == 0):
            # first image (with well names), bbox slightly shifted
            bbox = [0, height, width_tl, height]
        else:
            # other images
            bbox = [x_offset + width * _ci, height * (rows - (_ri + 1)), width, height]

        # get location of subplot for camera
        ax = axs[_loc]

        # plot first frame of video + annotate wells
        if fov:
            fovsplitter = FOVMultiWellsSplitter(ch_featfilepath)
            # let's check if the fovsplitter managed to get a frame automatically
            if fovsplitter.img is None:
                fovsplitter.img = get_frame_from_raw(feat2raw(ch_featfilepath))
            img_shape = fovsplitter.img_shape

            fovsplitter.plot_wells(is_rotate180=rotate, ax=ax, line_thickness=10)
        if not fov:
            vid = selectVideoReader(str(Path(str(ch_featfilepath).replace('Results','RawVideos')).parent.joinpath('metadata.yaml')))
            status, img = vid.read_frame(0)
            img_shape = img.shape
            plot_6wellplate(img, is_rotate180=rotate, ax=ax, line_thickness=10)
        
        # plot worm trajectories
        plot_trajectory(ch_featfilepath,
                       ax=ax,
                       downsample=downsample,
                       legend=False,
                       rotate=rotate,
                       img_shape=img_shape)

        # set image position in figure
        ax.set_position(bbox)

    plt.show()
    if saveDir:
        saveName = Path(featurefilepath).parent.stem + '.png'
        savePath = Path(saveDir) / saveName
        fig.savefig(savePath,
                    bbox_inches='tight',
                    dpi=300,
                    pad_inches=0,
                    transparent=True)
    return(fig)


def plot_plate_trajectories_from_filenames_summary(filenames_path, saveDir):
    """ Plot plate trajectories for all files in Tierpsy filenames summaries
        'filenames_path', and save results to 'saveDir' """

    from tierpsytools.read_data.hydra_metadata import _get_filename_column

    filenames_df = pd.read_csv(filenames_path, comment='#')
    filenames_list = filenames_df[filenames_df['is_good']==True][_get_filename_column(filenames_path)]

    filestem_list = []
    featurefile_list = []
    for fname in filenames_list:
        # obtain file stem
        filestem = Path(fname).parent.parent / Path(fname).parent.stem

        # only record featuresN filepaths with a different file stem as we only
        # need 1 camera's video per plate to find the others
        if filestem not in filestem_list:
            filestem_list.append(filestem)
            featurefile_list.append(fname)

    # overlay trajectories and combine plots for each plate that was imaged
    for featurefilepath in tqdm(featurefile_list):
        plot_plate_trajectories(featurefilepath, saveDir)


#%% Main

if __name__ == "__main__":
    print("\nRunning: ", sys.argv[0])

    #example_featuresN = Path("/Volumes/behavgenom$/Saul/MicrobiomeScreen96WP/Results/20200222/microbiome_screen2_run7_p1_20200222_122858.22956805/metadata_featuresN.hdf5")
    example_featuresN = Path('/Volumes/behavgenom$/John/data_exp_info/optimisation/condensation/Results/20240111/240111_avoidance_screen_condensation_run1_0000_20240111_161922.22956809/metadata_featuresN.hdf5')
    parser = argparse.ArgumentParser()
    # default to example file if none given
    parser.add_argument("--input", help="input file path (featuresN)",
                        default=example_featuresN)
    # default to input's grandparent if none given
    known_args = parser.parse_known_args()
    parser.add_argument("--output", help="output directory path (for saving)",
                        default=Path(known_args[0].input).parent.parent)
    # parser.add_argument("--downsample", help="downsample trajectory data by plotting the worm centroid for every 'nth' frame",
    #                     default=10)
    args = parser.parse_args()
    print("Input file:", args.input)
    print("Output directory:", args.output)

<<<<<<< HEAD
    plot_plate_trajectories(args.input, saveDir=args.output, downsample=10)
=======
    plot_plate_trajectories(args.input, saveDir=args.output, downsample=10, fov=False)


>>>>>>> upstream/master
