#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:03:30 2020

@author: ibarlow

Subsample the .mjpg files from the venoms data to test tierpsy paramenters

"""

import cv2
from pathlib import Path
import pandas as pd


def subsample_mjpg(fname,
                   outfile,
                   frame_rate=25,
                   subsample_size=4,
                   imgsize=(2048, 2048)):
    """
    
    little function that subsamples .mjpg videos and output as
    .avi to test tierpsy parameters

    Parameters
    ----------
    fname : string or pathname
        DESCRIPTION.
    outfile : string or pathname
        DESCRIPTION.
    frame_rate : frames per second
        DESCRIPTION.
    subsample_size : in minutes
        DESCRIPTION.
    imgsize : size per frame in pixels
        DESCRIPTION.

    Returns
    -------
    None.

    """

    subsample_frames = subsample_size * 60 * frame_rate

    vidcap = cv2.VideoCapture(str((fname)))

    outvid = cv2.VideoWriter(str(outfile),
                            cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                            frame_rate,
                            imgsize,
                            0)

    count = 0
    while vidcap.isOpened():

        if count < subsample_frames:
            # Capture frame-by-frame
            ret, frame = vidcap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            outvid.write(gray)


            count += 1
        else:
            vidcap.release()
    return


#%% EXAMPLE


if __name__ == '__main__':

    # input a .csv that contains files that you want to subsample
    file_list = Path('/Volumes/behavgenom$/Ida/Data/Phenix/venom_subsampling/venom_outliers.csv')
    fnames = pd.read_csv(file_list)
    
    fnames['fname'] = [r.fname.replace('hdf5', 'mjpg'
                                       ).replace('MaskedVideos_20200513', 'RawVideos')
                       for i,r in fnames.iterrows()]
    # add in an extra column that specifies the output file
    fnames['outfile'] = [file_list.parent / 
                         str(Path(r.fname).stem + 'subsample.avi') 
                         for i, r in fnames.iterrows()]

    for i, r in fnames.iterrows():
        print(i)
        
        try:
            if not r.outfile.exists():
                subsample_mjpg(r.fname, r.outfile)
            else:
                print('{} video already exists'.format(r.outfile))
        except Exception as error:
            print (error)

    # extra = Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/Ilastik/venoms/RawVideos/Control_set1(l.sin,A.lob)/Control_set1(l.sin,A.lob)_Ch1_08082019_133453.mjpg')
    # out_extra = extra.parent / str(extra.stem + 'subsample.avi')

    # extra = pd.Series(data=[str(extra), str(out_extra)],
    #                      index = ['fname', 'outfile'])
    # extra = extra.to_frame().transpose()

    # for i, r in extra.iterrows():
    #     print(i)
    #     try:
    #         subsample_mjpg(r.fname, r.outfile)
    #     except Exception as error:
    #         print (error)

