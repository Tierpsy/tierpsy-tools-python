#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 17:52:32 2021

@author: lferiani
"""
# %%
import cv2
import fire
import base64
import tables
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from io import BytesIO, StringIO
from pathlib import Path
from matplotlib import pyplot as plt

from tierpsytools.hydra import CAM2CH_df
from tierpsytools.hydra.read_imgstore_extradata import ExtraDataReader


def fix_dtypes(df):

    cols = df.columns

    # these are numeric-looking columns that should be strings without decimal
    strcols = [c for c in cols if ('serial' in c) or ('date' in c)]
    # these shoul dbe integer numbers
    intcols = [c for c in cols if ('number' in c) or ('slot' in c)]

    for c in strcols:
        if df[c].isna().all():
            continue
        try:
            df[c] = df[c].astype(int).astype(str)
        except Exception as e:
            raise Exception(
                'failed to convert - check for nans').with_traceback(
                    e.__traceback__)
    for c in intcols:
        if df[c].isna().all():
            continue
        try:
            df[c] = df[c].astype(int)
        except Exception as e:
            raise Exception(
                'failed to convert - check for nans').with_traceback(
                    e.__traceback__)

    return df


def get_project_root_from_meta_fname(input_path):
    assert 'AuxiliaryFiles' in str(input_path), 'AuxiliaryFiles not in path!'
    proj_root = str(input_path).split('AuxiliaryFiles')[0]
    return Path(proj_root)


def load_ch1_metadata(meta_fname, proj_root, is_strict=True):
    """
    load metadata, clean it up a little, add channel info
    """
    masked_dir = proj_root / 'MaskedVideos'
    raw_dir = proj_root / 'RawVideos'

    if is_strict:
        meta_df = pd.read_csv(meta_fname)
        meta_df = fix_dtypes(meta_df)
    else:
        meta_df = pd.read_csv(meta_fname)
        # allow metadata to have issues like missing videos or camera names
        is_problematic = meta_df['camera_serial'].isna()
        is_problematic = is_problematic | meta_df['imgstore_name'].isna()
        meta_df = meta_df[~is_problematic].reset_index()
        meta_df['camera_serial'] = (
            meta_df['camera_serial'].astype(int).astype(str))

    # merge as I want the channel number
    premerge_shape = meta_df.shape
    meta_df = meta_df.merge(CAM2CH_df, on='camera_serial', how='left')

    # checks
    assert meta_df.shape[0] == premerge_shape[0]
    assert meta_df.shape[1] == premerge_shape[1] + 2
    assert meta_df['rig'].equals(meta_df['instrument_name']), (
        "Merging went wrong: check camera string isn't a float in metadata")

    # isolate channel 1
    ch1_df = meta_df.query('channel == "Ch1"')[
        ['date_yyyymmdd', 'imaging_run_number',
         'imgstore_name', 'imaging_plate_id']
    ].sort_values(by='imgstore_name').drop_duplicates().reset_index(drop=True)

    ch1_df['masked_videos_name'] = [
        masked_dir / f / 'metadata.hdf5' for f in ch1_df['imgstore_name']]

    ch1_df['raw_videos_name'] = [
        raw_dir / f / 'metadata.yaml' for f in ch1_df['imgstore_name']]

    ch1_df = ch1_df.sort_values(by=['imgstore_name'])

    return ch1_df


def postprocess(img):
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 25)
    return


def get_plate_id_from_frame(frame):
    scaling_factor = 0.05
    # crop and resize
    physical_plate_id = np.rot90(frame[:, min(frame.shape):])
    height, width = physical_plate_id.shape
    height = int(scaling_factor * height)
    width = int(scaling_factor * width)
    physical_plate_id = cv2.resize(
        physical_plate_id, (width, height), interpolation=cv2.INTER_AREA)
    # physical_plate_id = 255 - physical_plate_id

    return physical_plate_id


def get_frame_from_masked(masked_fname):

    # read frame data
    with tables.File(masked_fname, 'r') as fid:
        frame = fid.get_node('/full_data')[0]

    return frame


def get_frame_from_raw(extradatareader_instance):

    frame = extradatareader_instance.store.get_next_image()[0]
    extradatareader_instance.store.close()

    return frame


def encode_img_for_html(img_array):
    is_success, buffer = cv2.imencode(".png", img_array)
    io_buf = BytesIO(buffer)
    img_str = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    img_str = 'data:image/png;charset=utf-8;base64,' + img_str
    img_str = f'<img src="{img_str}" />'
    return img_str


def encode_fig_for_html(fig):
    figfile = StringIO()
    plt.savefig(figfile, format='svg', bbox_inches='tight')
    figdata_svg = '<svg' + figfile.getvalue().split('<svg')[1]
    return figdata_svg


def get_row_info(row, is_img_postprocessing=True):

    # get info from raw video
    edr = ExtraDataReader(row.raw_videos_name)

    # frame
    try:
        frame = get_frame_from_raw(edr)
    except:
        # gets frame from masked as backup
        frame = get_frame_from_masked(row['masked_videos_name'])
    # now crop and scale
    img = get_plate_id_from_frame(frame)
    # postprocess the cropped bit
    if is_img_postprocessing:
        img = postprocess(img)
    # enconde
    img_str = encode_img_for_html(img)

    # sensors
    ed_df = edr.get_extra_data(
        includeonly=['light', 'tempi', 'tempo']).set_index('time')
    # light plot
    fig, ax = plt.subplots(figsize=(4, 1.2))
    ed_df['light'].plot(
        ax=ax, ylim=(0.1, 9000), logy=True)
    ax.set_ylabel('light')
    fig.tight_layout()
    light_str = encode_fig_for_html(fig)
    # temp plot
    ax.clear()
    ed_df['tempi'].plot(ax=ax, ylim=(19, 30), color='tab:orange')
    ax.set_ylabel(u'T$_i$, \u00B0C', color='tab:orange')
    ax2 = ax.twinx()
    ed_df['tempo'].plot(ax=ax2, ylim=(17, 26), color='tab:blue')
    ax2.set_ylabel(u'T$_o$, \u00B0C', color='blue')
    if (ed_df['tempi'] > 26).any():
        ax.set_facecolor('red')

    fig.tight_layout()
    temp_str = encode_fig_for_html(fig)

    plt.close(fig)

    return (
        row.imgstore_name, row.imaging_plate_id, img_str, light_str, temp_str)


def write_header(out_fname):
    with open(out_fname, 'w') as fid:
        print('<!DOCTYPE html>', file=fid)
        print('<html>', file=fid)
        print('<head>', file=fid)
        print('  <title>Plate Checker Report</title>', file=fid)
        print('  <style>', file=fid)
        print('  table, th, td {', file=fid)
        print('    font-family: Arial;', file=fid)
        print('    border: 1px solid black;', file=fid)
        print('    border-collapse: collapse;', file=fid)
        print('    text-align: center;}', file=fid)
        print('  </style>', file=fid)
        print('</head>', file=fid)
        print('<body>', file=fid)
        print('  <table style="width:100%">', file=fid)
        print('    <tr>', file=fid)
        print('      <th>imgstore_name</th>', file=fid)
        print('      <th>imaging_plate_id</th>', file=fid)
        print('      <th>real plate id</th>', file=fid)
        print('      <th>light intensity</th>', file=fid)
        print('      <th>temperature</th>', file=fid)
        print('    </tr>', file=fid)
    return


def write_table_row(out_fname, row_info_tuple):
    imgstore_name, imaging_plate_id, img_str, light_str, temp_str = (
        row_info_tuple)
    with open(out_fname, 'a') as fid:
        print('    <tr>', file=fid)
        print(f'      <td>{imgstore_name}</td>', file=fid)
        print(f'      <td>{imaging_plate_id}</td>', file=fid)
        print(f'      <td>{img_str}</td>', file=fid)
        print(f'      <td>{light_str}</td>', file=fid)
        print(f'      <td>{temp_str}</td>', file=fid)
        print('    </tr>', file=fid)
    return


def write_closing_tags(out_fname):
    with open(out_fname, 'a') as fid:
        print('  </table>', file=fid)
        print('</body>', file=fid)
        print('<html>', file=fid)
    return


# def process_row(metadata_row):
#     metadata_row, out_fname = metadata_row
#     row_tuple = get_row_info(metadata_row)
#     write_table_row(out_fname, row_tuple)
#     return


def _platechecker(
        metadata_path, project_root=None, output_path=None,
        is_img_postprocessing=True):
    """
    platechecker
    Scan the metadata provided, and create a html report with
    sensor data, and a crop of the first frame showing the physical id tag
    of the imaging plate.
    If your project is using the standard folder structure, you can just use
    metadata_path as an input, and project_root and output will be inferred.

    Parameters
    ----------
    metadata_path : Path or str
        path to the metadata csv file
    project_root : Path or str, optional
        Root folder for the project. Should contain RawVideos, MaskedVideos,
        Results, AuxiliaryFiles. By default None => it will be inferred from
        the metadata path
    output_path : Path or str, optional
        Path to the output html file. If omitted,
        it will be `project_root/AuxiliaryFiles/platechecker_report.html`
    is_img_postprocessing : logical, optional, default True
        If True, the part of frame where the imaging plate id is written
        will be enhanced. Hopefully this makes it easier to read. If not,
        set it to False
    """

    if isinstance(metadata_path, str):
        metadata_path = Path(metadata_path)

    # paths
    if project_root is None:
        project_root = get_project_root_from_meta_fname(metadata_path)
    else:
        if isinstance(project_root, str):
            project_root = Path(project_root)
    if output_path is None:
        aux_dir = project_root / 'AuxiliaryFiles'
        output_path = aux_dir / 'platechecker_report.html'
    else:
        if isinstance(output_path, str):
            output_path = Path(output_path).with_suffix('.html')

    # load metadata
    meta_df = load_ch1_metadata(
        metadata_path, proj_root=project_root, is_strict=True)

    # start writing to the output file
    write_header(output_path)

    # loop and write
    for row in tqdm(meta_df.itertuples()):
        row_data_out = get_row_info(
            row, is_img_postprocessing=is_img_postprocessing)
        write_table_row(output_path, row_data_out)

    write_closing_tags(output_path)
    return


def platechecker():
    fire.Fire(_platechecker)


if __name__ == '__main__':

    proj_root = Path('/Volumes/behavgenom$/Ida/Data/Hydra/PrestwickScreen/')
    meta_fname = proj_root / (
        'AuxiliaryFiles/20210210wells_updated_metadata_nonan.csv')
    platechecker(meta_fname)
