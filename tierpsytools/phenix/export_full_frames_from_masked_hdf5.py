#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:40:25 2020

@author: ibarlow

script to export the full frames from the masked videos


"""

import cv2
import tables
from pathlib import Path

#%%

def extract_full_frames(masked_hdf5):
    """
    Opens masked video hdf and file and saves the full frames into a
    new directory called 'full_frames'

    Parameters
    ----------
    masked_hdf5 : path to masked_video.hdf5
        DESCRIPTION. from tierpsy

    Returns
    -------
    None.

    """

    SAVE_DIR = masked_hdf5.parent / '{}_full_frames'.format(masked_hdf5.stem)

    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir()

    with tables.File(masked_hdf5, 'r') as fid:
        dims = fid.get_node('/full_data').shape

        for count, file in enumerate(range(0, dims[0])):
            img = fid.get_node('/full_data')[file]
            outfile = str(SAVE_DIR) + '/{}_fullframe.tif'.format(count)

            cv2.imwrite(outfile,
                        img)

    return


#%% EXAMPLE
if __name__ == '__main__':

    MASKED_DIR = Path('/Volumes/SAMSUNG/venoms/MaskedVideos')

    GET_SETS = [d for d in MASKED_DIR.glob('*') if d.is_dir()]

    for subdir in GET_SETS:
        fnames = list(subdir.glob('*.hdf5'))

        for f in fnames:
            print('Extracting full frames from: {}'.format(fname))
            extract_full_frames(f, resize=False)
