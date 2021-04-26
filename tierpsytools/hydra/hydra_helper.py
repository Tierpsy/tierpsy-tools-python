#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:29:20 2019

@author: em812
"""
import re
import numpy as np
import pandas as pd
import warnings

def get_camera_serial(
        metadata, n_wells=96
        ):
    """
    @author: em812
    Get the camera serial number from the well_name and instrument_name.

    param:
        metadata: pandas dataframe
            Dataframe with day metadata

    return:
        out_metadata: pandas dataframe
            Day metadata dataframe including camera serial

    """
    from tierpsytools.hydra import CAM2CH_df,UPRIGHT_96WP

    if n_wells != 96:
        raise ValueError('Only 96-well plates supported at the moment.')

    channels = ['Ch{}'.format(i) for i in range(1,7,1)]

    WELL2CH = []
    for ch in channels:
        chdf = pd.DataFrame(UPRIGHT_96WP[ch].values.reshape(-1,1),
                            columns=['well_name'])
        chdf['channel'] = ch
        WELL2CH.append(chdf)
    WELL2CH = pd.concat(WELL2CH,axis=0)

    WELL2CAM = pd.merge(
            CAM2CH_df,WELL2CH,
            how='outer',on='channel'
            ).sort_values(by=['rig','channel','well_name'])
    # keep only the instruments that exist in the metadata
    WELL2CAM = WELL2CAM[WELL2CAM['rig'].isin(metadata['instrument_name'])]

    # Rename 'rig' to 'instrument_name'
    WELL2CAM = WELL2CAM.rename(columns={'rig':'instrument_name'})

    # Add camera number to metadata
    out_metadata = pd.merge(
            metadata,WELL2CAM[['instrument_name','well_name','camera_serial']],
            how='outer',left_on=['instrument_name','well_name'],
            right_on=['instrument_name','well_name']
            )
    if not out_metadata.shape[0] == metadata.shape[0]:
        raise Exception('Wells missing from plate metadata.')

    if not all(~out_metadata['camera_serial'].isna()):
        raise Exception('Camera serial not found for some wells.')

    return out_metadata


def add_imgstore_name(
        metadata, raw_day_dir, n_wells=96, run_number_regex=r'run\d+_'
        ):
    """
    @author: em812
    Add the imgstore name of the hydra videos to the day metadata dataframe.

    param:
        metadata = pandas dataframe
            Dataframe with metadata for a given day of experiments.
            See README.md for details on fields.
        raw_day_dir = path to directory
            RawVideos root directory of the specific day, where the
            imgstore names can be found.
        n_wells = integer
            Number of wells in imaging plate (only 96 supported at the
            moment)

    return:
        out_metadata = metadata dataframe with imgstore_name added

    """
    from os.path import join
    from tierpsytools.hydra.hydra_helper import run_number_from_regex

    ## Checks
    # - check if raw_day_dir exists
    if not raw_day_dir.exists:
        warnings.warn("\nRawVideos day directory was not found. "
                      +"Imgstore names cannot be added to the metadata.\n",
                      +"Path {} not found.".format(raw_day_dir))
        return metadata

    # - if the raw_dat_dir contains a date in yyyymmdd format, check if the
    #   date in raw_day_dir matches the date of runs stored in the metadata
    #   dataframe
    date_of_runs = metadata['date_yyyymmdd'].astype(str).values[0]
    date_in_dir = re.findall(r'(20\d{6})',raw_day_dir.stem)
    if len(date_in_dir)==1 and date_of_runs != date_in_dir[0]:
        warnings.warn(
            '\nThe date in the RawVideos day directory does not match ' +
            'the date_yyyymmdd in the day metadata dataframe. '
            'Imgstore names cannot be added to the metadata.\n'+
            'Please check the dates and try again.')
        return metadata

    # add camera serial number to metadata
    metadata = get_camera_serial(metadata, n_wells=n_wells)

    # get imgstore full paths = raw video directories that contain a
    # metadata.yaml file and get the run and camera number from the names
    file_list = [file for file in raw_day_dir.rglob("metadata.yaml")]
    #print('There are {} raw videos found in {}.\n'.format(
    #    len(file_list),raw_day_dir))
    camera_serial = [str(file.parent.parts[-1]).split('.')[-1]
                               for file in file_list]

    imaging_run_number = run_number_from_regex(file_list, run_number_regex)
    # imaging_run_number = run_number_from_timestamp(file_list, camera_serial)

    file_meta = pd.DataFrame({
        'file_name': file_list,
        'camera_serial': camera_serial,
        'imaging_run_number': imaging_run_number
        })

    # keep only short imgstore_name (experiment_day_dir/imgstore_name_dir)
    file_meta['imgstore_name'] = file_meta['file_name'].apply(
            lambda x: "/".join(x.parts[-3:-1]))

    # merge dataframes to store imgstore_name for each metadata row
    out_metadata = pd.merge(
            metadata,
            file_meta[['imaging_run_number','camera_serial','imgstore_name']],
            how='outer',on=['imaging_run_number','camera_serial'])

    ## Checks
    # - check if there are missing videos (we expect to have videos from every
    #   camera of a given instrument). If yes, raise a warning.
    if out_metadata['imgstore_name'].isna().sum()>0:
        not_found = out_metadata.loc[out_metadata['imgstore_name'].isna(),
                                 ['imaging_run_number', 'camera_serial']]
        for i,row in not_found.iterrows():
            warnings.warn('\n\nNo video found for day '
                          +'{}, run {}, camera {}.\n\n'.format(
                                  raw_day_dir.stem,*row.values)
                          )

    return out_metadata


def get_date_of_runs_from_aux_files(manual_metadata_file):
    """
    @author: em812
    Finds the date of the runs in the manual_metadata_file.
    If the date field is missing, then it looks for the date of runs in the
    manual_metadata_file file name (in the format yyyymmdd).
    If there is no date in this format in the file name, then it looks at the
    folder name (which is the folder for a specific day of experiments).
    If the date in yyyymmdd format cannot be found in any of these locations,
    an error is raised.

    param:
        manual_metadata_file: full path to .csv file
            Full path to the manual metadata file

    return:
        date_of_runs: string
            The date of the experiments, in yyyymmdd format
    """
    manual_metadata = pd.read_csv(manual_metadata_file, index_col=False)
    if 'date_yyyymmdd' in manual_metadata.columns:
        date_of_runs = manual_metadata['date_yyyymmdd'].astype(str).values[0]
    else:
        date_of_runs = re.findall(r'(20\d{6})',manual_metadata_file.stem)
        if len(date_of_runs)==1:
            date_of_runs = date_of_runs[0]
        else:
            date_of_runs = re.findall(r'(20\d{6})',
                                      manual_metadata_file.parent.stem)
            if len(date_of_runs)==1:
                date_of_runs = date_of_runs[0]
            else:
                raise ValueError('The date of the experiments cannot be '
                                 +'identified in the auxiliary files path '
                                 +'names. Please add a data_yyyymmdd column '
                                 +'to the manual_metadata file.'
                                 )

    # If the aux_day_dir contains the date, then make sure it matches the date
    # extracted from the manual_metadate_file
    date_in_dir = re.findall(r'(20\d{6})',manual_metadata_file.parent.stem)
    if len(date_in_dir)==1 and date_in_dir[0]!=date_of_runs:
        raise ValueError('\nThe date_of_runs taken from the '
                         +'manual_metadata_file ({}) '.format(date_of_runs)
                         +'does not match the date of runs in the folder '
                         +'name {}.\n'.format(manual_metadata_file.parent)
                         +'Please set the correct date and try again.')
    return date_of_runs


def convert_bad_wells_lut(bad_wells_csv):
    """
    author: @ilbarlow
    Function for converting bad_wells_csv for input into dataframes
    Input:
    bad_wells_csv -.csv file listing imaging_plate_id and well_name

    Output:
    bad_wells_df - DataFrame with columns imaging_plate_id, well_name and
    is_bad_well=True
    """

    bad_wells_df = pd.read_csv(bad_wells_csv)
                                         # encoding='utf-8-sig')
    bad_wells_df['is_bad_well'] = True

    return bad_wells_df

def column_from_well(wells_series):
    return wells_series.apply(lambda x: re.findall(r'\d+', x)[0]).astype(int)

def row_from_well(wells_series):
    return wells_series.apply(lambda x: re.findall(r'[^0-9]', x)[0])

def exract_randomized_by(robotlog,randomized_by):
    if 'column' in randomized_by:
        # extract the column number from the well number for mapping (as replicates are by column)
        robotlog[randomized_by] = column_from_well(robotlog['source_well'])
    elif 'row' in randomized_by:
        # extract the row number from the well number for mapping (if replicates are by row)
        robotlog[randomized_by] = row_from_well(robotlog['source_well'])
    elif ('well' in randomized_by) and (randomized_by != 'source_well'):
        robotlog[randomized_by] = robotlog['source_well']
    return robotlog

def rename_out_meta_cols(out_meta,randomized_by):

    out_meta = out_meta.rename(columns={'column':'source_plate_column',
                                            'row':'source_plate_row'})
    if 'column' in randomized_by:
        out_meta.rename(columns={randomized_by:'source_plate_column'})
    elif 'row' in randomized_by:
        out_meta.rename(columns={randomized_by:'source_plate_row'})
    elif ('well' in randomized_by) and (randomized_by != 'source_well'):
        out_meta = out_meta.drop(columns=[randomized_by])
    return out_meta

def explode_df(df, lst_col, fill_value='', preserve_index=False):

    # make sure `lst_cols` is list-like
    if (lst_col is not None
        and len(lst_col) > 0
        and not isinstance(lst_col, (list, tuple, np.ndarray, pd.Series))):
        lst_col = [lst_col]
    # all columns except `lst_col`
    idx_cols = df.columns.difference(lst_col)
    # calculate lengths of lists
    lens = df[lst_col[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_col}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res

def run_number_from_regex(file_list, run_number_regex=r'run\d+_'):
    """
    Get the run number from the name of the raw video based on a regex expression
    """
    run_name = [re.findall(run_number_regex, str(file.parent.parts[-1]))
                for file in file_list]
    run_name = [x[0] if len(x)==1 else '' for x in run_name ]
    imaging_run_number = [re.findall(r'\d+', x) for x in run_name]
    imaging_run_number = [int(x[0]) if len(x)==1 else np.nan for x in imaging_run_number]
    if np.any(np.isnan(imaging_run_number)):
        raise Exception('Run number could not be recovered from all raw video imgstore names.')
    return imaging_run_number

def run_number_from_timestamp(file_list, camera_serial):
    """
    Get the run number based on the camera serial and the timestamp.
    !! This function is not safe, so the option to get the run number based
    in the timestamp is not integrated to the standard functions to
    compile metadata for hydra
    """
    timestamp = [int(file.parent.stem.split('_')[-1]) for file in file_list]
    bluelight = []
    for file in file_list:
        blue = [b for b in ['bluelight', 'prestim', 'poststim'] if b in file.parent.stem]
        if len(blue)==1:
            blue = blue[0]
        elif len(blue)==0:
            blue = 'nobluelight'
        elif len(blue)>1:
            raise Exception('More than one bluelight label in video {}.'.format(file))
        bluelight.append(blue)

    df = pd.DataFrame({
        'bluelight': bluelight,
        'file_name':file_list,
        'camera_serial': camera_serial,
        'timestamp': timestamp
        })

    df = df.groupby(['camera_serial', 'bluelight']).apply(
        lambda x: x.sort_values(by=['timestamp'], ascending=True).assign(
            imaging_run_number=list(range(1,x.shape[0]+1))
            ))

    df = df.reset_index(level=2).sort_values(by='level_2')
    assert all([x==y for x,y in zip(file_list, df['file_name'].to_list())])

    return df['imaging_run_number'].to_list()


def check_dates_in_yaml(metadata,raw_day_dir):
    """
    Checks the day metadata, to make sure that the experiment date stored in
    the metadata dataframe matches the date in the metadata.yaml file in the
    corresponding raw video directory.

    param:
        metadata : pandas dataframe
            Dataframe containing all the metadata from one day of experiments
        raw_day_dir : directory path
            Path of the directory containing the RawVideos for the
            specific day of experiments.

    return:
        None
    """

    return


if __name__=="__main__":
    df = pd.DataFrame([[0,1],[1,2]],columns=['a','b'])
    df['c'] = pd.Series([[0],[0,1]])
    df['d'] = pd.Series([[],[0,1]])
    df1 = explode_df(df,['c'])
    print(df1)
    df2 = explode_df(df,['d'])
    print(df2)