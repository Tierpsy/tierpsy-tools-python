#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:57:10 2019

@author: lferiani
"""
import pandas as pd
from pathlib import Path
from tierpsytools.hydra import CAM2CH_df


def raw_to_masked(fname):
    out = str(fname).replace("RawVideos", "MaskedVideos")
    out = Path(out.replace(".yaml", ".hdf5"))
    return out


def raw_to_featuresN(fname):
    out = str(fname).replace("RawVideos", "Results")
    out = Path(out.replace(".yaml", "_featuresN.hdf5"))
    return out


def find_matching_tierpsy_files(project_dir, is_return_relative_path=False):
    """find_matching_tierpsy_files

    Scan the folder 'project_dir / RawVideos' looking for imgstore files.
    Predict the names that masked videos and featureN files would have.
    Return a list of matching raw video, masked video, featuresN paths.

    Parameters
    ----------
    project_dir : str or path
        This is the project directory. It is assumed that it contains a
        RawVideos folder.
    is_return_relative_path : boolean, optional
        If True, return the paths to raw videos, masked videos,
        and features files as relative paths starting from project_dir.
        The default is False.

    Returns
    -------
    matched_fnames : list of tuples of path objects
        List of tuples (rawvideo_path, maskedvideo_path, featuresN_path).

    """
    # input check
    if isinstance(project_dir, str):
        project_dir = Path(project_dir)
    # get dir name
    rawvideos_dir = project_dir / "RawVideos"
    # find list of files
    rawvideos_fnames = list(rawvideos_dir.rglob("metadata.yaml"))
    # loop on files, get the name of corresponding masked and featuresN files
    # and check their existance
    matched_fnames = []
    # unmatched_fnames = []
    unmatched_counter = 0
    # loop on raw videos
    for rv in rawvideos_fnames:
        fv = raw_to_featuresN(rv)
        if not fv.exists():
            # unmatched_fnames.append(rv)
            unmatched_counter += 1
        else:
            mv = raw_to_masked(rv)
            if mv.exists():
                matched_fnames.append((rv, mv, fv))
            else:
                # unmatched_fnames.append(rv)
                unmatched_counter += 1

    if unmatched_counter != 0:
        print(
            "Couldn't match {} videos found in {}".format(
                unmatched_counter, rawvideos_dir
            )
        )

    if is_return_relative_path:
        matched_fnames = [
            (
                rv.relative_to(project_dir),
                mv.relative_to(project_dir),
                fv.relative_to(project_dir),
            )
            for (rv, mv, fv) in matched_fnames
        ]

    return matched_fnames


def parse_camera_serial(filename):
    import re

    regex = r"(?<=20\d{6}\_\d{6}\.)\d{8}"
    camera_serial = re.findall(regex, str(filename).lower())[0]
    return camera_serial


def find_imgstore_videos(target):
    """
    find_imgstore_videos scans the target directory, or the target files list,
    and returns a dataframe with information about the videos found.

    Parameters
    ----------
    target : str or pathlib.Path
        path to the directory cointaining multiple imgstore files or to a file
        containing a list thereof.

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing, for each imgstore file:
            1) full_path
            2) imgstore (the name of the folder containing the metadata.yaml)
            3) imaging_set (imgstore stripped of camera serial number)
            4) camera_serial (unique id of camera)
            5) channel (Ch1 to Ch6, position of camera in the rig)
            6) rig (Hydra01 to Hydra05)

    """

    # input check
    if isinstance(target, str):
        target = Path(target)

    # find all yamls
    if target.is_dir():
        imgstore_yamls = list(target.rglob("metadata.yaml"))
    elif target.is_file():
        with open(target, 'r') as fid:
            imgstore_yamls = [Path(yaml) for yaml in fid.read().splitlines()]
    else:
        TypeError('Target needs to be the path to a directory or a file.')


    # now loop on imgstore yamls and return:
    # 1) the parent directory
    # 2) the imaging set name (parent directory - camera number)
    # 3) the camera number
    imgstore_dirs = []
    imagingset_names = []
    camera_serials = []
    for yaml in imgstore_yamls:
        imgstore_dir = yaml.parent.name
        serial = parse_camera_serial(imgstore_dir)
        setname = "".join(imgstore_dir.split("." + serial)[:-1])
        imgstore_dirs.append(imgstore_dir)
        camera_serials.append(serial)
        imagingset_names.append(setname)
    # format into df, then merge with df serial -> channel, rig
    df = pd.DataFrame(
        zip(imgstore_yamls, imgstore_dirs, imagingset_names, camera_serials),
        columns=["full_path", "imgstore", "imaging_set", "camera_serial"],
    )
    df = df.merge(CAM2CH_df, how="left", on="camera_serial")

    return df


if __name__ == "__main__":

    # set folders
    wd = Path("/Volumes/behavgenom$/Ida/Data/Hydra/ICDbacteria")
    matched = find_matching_tierpsy_files(wd)
