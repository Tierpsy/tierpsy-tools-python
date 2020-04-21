#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:25:41 2019

@author: em812
"""
import pandas as pd
from pathlib import Path
from tierpsytools.hydra.hydra_helper import exract_randomized_by
from tierpsytools.hydra.hydra_helper import rename_out_meta_cols, explode_df
import re
import warnings
import numpy as np
import itertools

#%%
def merge_robot_metadata(
        sourceplates_file, randomized_by='column', saveto=None,
        del_if_exists=False
        ):
    """
    @author: em812
    Function that imports the robot runlog and the associated source plates
    for a given day of experiments and uses them to compile information of
    drugs in the destination plates.

    param:
        sourceplates_file: path to .csv file
            path to sourceplates_file `YYYYMMDD_sourceplates.csv`
        randomized_by: string
            How did the robot randomize the wells from the source
            plate to the destination plates?
            options: 'column'/'source_column' = shuffled columns,
                     'row'/'source_row' = shuffled rows,
                     'well'/'source_well' = shuffled well-by-well
            The parameter randomized_by is expected to be a field in the
            sourceplates file.
        saveto: path to .csv file
            The full path of the file where the robot metadata will be saved.
            If None, the robot metadata dataframe is not saved to disk.
        del_if_exists: boolean
            If True, then if the saveto file exists, it will be replaced.

    return:
        robot_metadata: pandas dataframe
            Robot related metadata for the given day of experiments as dataframe

    """

    if saveto is None:
        date = sourceplates_file.stem.split('_')[0]
        saveto = Path(sourceplates_file).parent / (date+'_robot_metadata.csv')

    # check if file exists
    if (saveto is not False) and (saveto.exists()):
        if del_if_exists:
            warnings.warn('Robot metadata file '
                          +'{} already exists. '.format(saveto)
                          +'File will be overwritten.')
            saveto.unlink()
        else:
            warnings.warn('Robot metadata file {} '.format(saveto)
                          +'already exists. Nothing to do here. If you want '
                          +'to recompile the robot metadata, rename or delete '
                          +'the exisiting file.')
            return

    # required fields in sourceplates file
    sourceplate_cols = ['source_plate_id', 'robot_runlog_filename',
                        'source_robotslot']+[randomized_by]

    # import the sourceplates
    sourceplates = pd.read_csv(sourceplates_file,index_col=False)

    # check if sourceplate dataframe fullfills requirements
    missing_cols = [col
                    for col in sourceplate_cols
                    if col not in sourceplates.columns]
    if len(missing_cols)>0:
        raise KeyError(
                'Field(s) {} '.format(missing_cols)
                +'do not exist in sourceplates file. This/These field(s) '
                +'is/are required.'
                )
    for plate in sourceplates['source_plate_id'].unique():
        runlogs = sourceplates.loc[sourceplates['source_plate_id']==plate,
                                   'robot_runlog_filename'].unique()
        if runlogs.shape[0]>1:
            raise ValueError('Multiple robot runlogs were defined for the same'
                             +'source plate.')

    # read each robot log and compile metadata
    robot_metadata = []
    for n_log,log_file in enumerate(
            sourceplates['robot_runlog_filename'].unique()):

        # get sourceplates lines linked to this robot log
        source = sourceplates.loc[
                sourceplates['robot_runlog_filename']==log_file,:]
        # get only unique rows of source_robotslot - source_plate_id
        source_map = source[
                ['source_robotslot','source_plate_id']
                ].drop_duplicates()
        # assert that a signle source_robotslot is defined for each single
        # source_plate_id
        assert source_map['source_plate_id'].unique().shape[0] == \
            source_map.shape[0] == \
            source_map['source_robotslot'].unique().shape[0]

        # read robotlog data
        robotlog = pd.read_csv(log_file)
        # keep only data for source_slots with drugs (drop water source_slots)
        robotlog = robotlog[
                robotlog['source_slot'].isin(source['source_robotslot'])
                ]
        # assign robot runlog id
        robotlog['robot_runlog_id'] = n_log+1
        # extract the column number or the row number from the well number for
        # mapping if the robot randomized based on columns or rows
        robotlog = exract_randomized_by(robotlog,randomized_by)
        # add source_plate_id based on unique source_plate_id - source_slot
        # mapping obtained from sourceplates file
        robotlog['source_plate_id'] = robotlog['source_slot'].map(
                dict(source_map.values))

        # merge all sourceplate data with robot runlog data based on
        # source_plate_id and randomized_by
        out_meta = pd.merge(
                source, robotlog, how='outer',
                left_on=['source_plate_id',randomized_by],
                right_on=['source_plate_id',randomized_by]
                )
        # get unique imaging_plate_id
        out_meta['imaging_plate_id'] = out_meta[
                ['source_plate_id','destination_slot']
                ].apply(
                    lambda x: 'rr{0}_sp{1}_ds{2}'.format(n_log+1,*x),axis=1
                    )

        # clean up and rename columns in out_meta
        # - sort rows for readability
        out_meta = out_meta.sort_values(
                by=['source_plate_id','destination_slot', randomized_by]
                ).reset_index(drop=True)
        # - drop duplicate source_slot info
        assert np.all(out_meta['source_slot']==out_meta['source_robotslot'])
        out_meta = out_meta.drop(labels='source_slot',axis=1)
        # - rename column field for interpretability
        out_meta = rename_out_meta_cols(out_meta,randomized_by)
        # - rearrange columns for readability
        leading_cols = ['imaging_plate_id', 'source_plate_id',
                        'destination_slot']
        end_cols = ['robot_runlog_id', 'robot_runlog_filename']
        out_meta = out_meta[
                leading_cols
                + list(out_meta.columns.difference(leading_cols + end_cols))
                + end_cols
                ]

        # append to list
        robot_metadata.append(out_meta)

    # concatenate data from all the robot runlogs
    robot_metadata = pd.concat(robot_metadata, axis=0)

    if saveto is not False:
        robot_metadata.to_csv(saveto, index=None)

    return robot_metadata

#%%

def populate_96WPs(worm_sorter,
                   entire_rows=True,
                   saveto=None,
                   del_if_exists=False
                   ):
    """
    @author: ilbarlow
    Populate 96 well plates for tracking experiment where plates have been
    filled with different strains of food or worms.
    param:
        worm_sorter: path for YYYYMMDD_wormsorter.csv file
    return:
        plate_metadata: one line per well; can be used in get_day_metadata
            function to compile with manual metadata
    """

    # parameters for the 96WPs
    n_columns = 12
    column_names = np.arange(1, n_columns+1)
    row_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    well_names = list(itertools.product(row_names, column_names))

    # saving and overwriting checks
    if saveto is None:
        date = worm_sorter.stem.split('_')[0]
        saveto = Path(worm_sorter).parent / (date+'_plate_metadata.csv')

    # check if file exists
    if (saveto is not False) and (saveto.exists()):
        if del_if_exists:
            warnings.warn('Plate metadata file {} already '.format(saveto)
                          + 'exists.File will be overwritten.')
            saveto.unlink()
        else:
            warnings.warn('Plate metadata file {} already '.format(saveto)
                          + 'exists. Nothing to do here. If you want to '
                          + 'recompile the day metadata, rename or delete the '
                          + 'exisiting file.')
            return None

    # import worm_sorter metadata and find the start and end rows and columns
    worm_sorter_df = pd.read_csv(worm_sorter)
    worm_sorter_df['start_row'] = [re.findall(r"[A-Z]", r.start_well)[0]
                                   for i, r in worm_sorter_df.iterrows()
                                   ]
    worm_sorter_df['end_row'] = [re.findall(r"[A-Z]", r.end_well)[0]
                                 for i, r in worm_sorter_df.iterrows()
                                 ]
    worm_sorter_df['start_column'] = [re.findall(r"(\d{1,2})", r.start_well)[0]
                                      for i, r in worm_sorter_df.iterrows()
                                      ]
    worm_sorter_df['end_column'] = [re.findall(r"(\d{1,2})", r.end_well)[0]
                                    for i, r in worm_sorter_df.iterrows()
                                    ]

    # create 96WP template to fill up
    plate_template = pd.DataFrame()
    plate_template['imaging_plate_row'] = [i[0] for i in well_names]
    plate_template['imaging_plate_column'] = [i[1] for i in well_names]
    plate_template['well_name'] = [i[0] + str(i[1]) for i in well_names]

    if not entire_rows:
        print('cannot use this function for generating the metadata;' +
              ' fills entire rows only.')
        return None

    # populate the dataframe
    plate_metadata = []
    for i, r in worm_sorter_df.iterrows():
        _section = (
                plate_template[
                        (r.start_row <= plate_template.imaging_plate_row)
                        & (plate_template.imaging_plate_row <= r.end_row)
                        ]).reset_index(drop=True)
        _details = pd.concat(
                _section.shape[0]*[r.to_frame().transpose()]
                ).reset_index(drop=True)
        plate_metadata.append(
                pd.concat([_section, _details], axis=1, sort=True))

    plate_metadata = pd.concat(plate_metadata)
    plate_metadata.to_csv(saveto, index=False)

    return plate_metadata

# %%extra function if using both robot and wormsorter


def merge_robot_wormsorter(day_root_dir,
                           robot_metadata,
                           plate_metadata,
                           bad_wells_csv,
                           merge_on=['imaging_plate_id', 'well_name'],
                           saveto=None,
                           del_if_exists=False
                           ):
    """
    author: @ilbarlow
    Function for combining the outputs of the wormsorter and robot so that
    concise and comprehensive dataframe can be used in get_day_metadata
    Input:
    day_root_dir - root directory of metadata
    robot_metadata - output from merge_robot_metadata function
    plate_metadata - output form populate_96WP function
    bad_wells_csv - .csv file listing imaging_plate_id and bad_well
    Ouput:
    concat_metadata - dataframe that can be used in get_day_metadata
    """

    # saving and overwriting checks
    if saveto is None:
        date = day_root_dir.stem
        saveto = day_root_dir / (date + '_merged_metadata.csv')

    # check if file exists
    if (saveto is not False) and (saveto.exists()):
        if del_if_exists:
            warnings.warn('Plate metadata file {} already '.format(saveto)
                          + 'exists. File will be overwritten.')
            saveto.unlink()
        else:
            warnings.warn('Merged robot and plate metadata file {} already '
                          .format(saveto) + 'exists. Nothing to do here.'
                          + 'If you want to recompile the plate metadata,'
                          + 'rename or delete the exisiting file.')
            return None

    robot_metadata.rename(columns={'destination_well': 'well_name'},
                          inplace=True)

    # parse out bad wells in robot_metadata into new column
    print('Finding any bad wells noted in the sourceplates and robot metadata')
    bad_source_lut = robot_metadata[robot_metadata['bad_wells'].notna()][
            ['bad_wells', 'imaging_plate_id']].drop_duplicates()

    if not bad_source_lut.empty:
        bad_well_lut = pd.concat([pd.Series(r['imaging_plate_id'],
                                            r['bad_wells'].split(','))
                                  for i, r in bad_source_lut.iterrows()]
                                 ).reset_index()
        bad_well_lut.rename(columns={'index': 'well_name',
                                     0: 'imaging_plate_id'},
                            inplace=True)
        bad_well_lut['is_bad_well'] = True

    else:
        print('No bad wells marked in the source plates file')
        bad_well_lut = pd.DataFrame()

    # read in the bad imaging wells - noted by Luigi during robot runs
    if bad_wells_csv:
        robot_bad_wells_df = pd.read_csv(bad_wells_csv,
                                         encoding='utf-8-sig')
        robot_bad_wells_df['is_bad_well'] = True
        bad_well_lut = pd.concat([robot_bad_wells_df,
                                  bad_well_lut], axis=0, sort=True
                                 ).reset_index(drop=True).drop_duplicates()
    else:
        print('no bad_wells.csv input to combine with robot metadata')

    # combine bad wells from both sources
    if not bad_well_lut.empty:
        print('Concatenating robot_metadata, plate_metadata and bad wells')
        concat_metadata = robot_metadata.set_index(merge_on).join(
                                        [plate_metadata.set_index(merge_on),
                                         bad_well_lut.set_index(merge_on)],
                                     how='outer')
        concat_metadata['is_bad_well'].fillna(False, inplace=True)

    else:
        print('No bad wells on this day of tracking; concatenating robot' +
              'and plate metadata')
        concat_metadata = robot_metadata.set_index(merge_on).join(
                plate_metadata.set_index(merge_on),
                how='outer')
        concat_metadata['is_bad_well'] = False

    concat_metadata.reset_index(drop=False,
                                inplace=True)
    # check that the number of output rows == input rows
    assert robot_metadata.shape[0] == concat_metadata.shape[0]

    concat_metadata.to_csv(saveto, index=False)

    return concat_metadata


#%%
# STEP 2
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
    assert out_metadata.shape[0] == metadata.shape[0]

    return out_metadata


def add_imgstore_name(
        metadata, raw_day_dir, n_wells=96
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
        warnings.warn('\nThe date in the RawVideos day directory does not '
                      +'match the date_yyyymmdd in the day metadata '
                      +'dataframe. Imgstore names cannot be added to the '
                      +'metadata.\nPlease check the dates and try again.')
        return metadata

    # add camera serial number to metadata
    metadata = get_camera_serial(metadata,n_wells=n_wells)

    # get imgstore full paths = raw video directories that contain a
    # metadata.yaml file and match the run and camera number
    MAP2PATH = metadata[
            ['imaging_run_number','camera_serial']
            ].drop_duplicates()
    MAP2PATH['run_name'] = metadata['imaging_run_number'].apply(
            lambda x: 'run{}'.format(x)
            )
    #print('There are {} videos expected from metadata in {}.\n'.format(
    #    MAP2PATH.shape[0],aux_day_dir))

    file_list = [file for file in raw_day_dir.rglob("metadata.yaml")]
    #print('There are {} raw videos found in {}.\n'.format(
    #    len(file_list),raw_day_dir))

    MAP2PATH['imgstore_name'] = MAP2PATH[
            ['run_name','camera_serial']
            ].apply(
                lambda x: [file
                           for file in file_list
                           if np.all([str(ix) in str(file)
                           for ix in x])],
                axis=1)

    ## Checks
    # - check if there are multiple videos with the same day, run and camera
    #   number. If yes, raise a warning (it is not necessarily an error, but
    #   good to let the user know).
    not_unique = MAP2PATH['imgstore_name'].apply(
            lambda x: True if len(x)>1 else False).values
    if np.sum(not_unique)>0:
        warnings.warn('\n\nThere is(are) {} set(s) of '.format(np.sum(not_unique))
                      +'videos with the same day, run and camera number.\n\n'
                     )
    # - check if there are missing videos (we expect to have videos from every
    #   camera of a given instrument). If yes, raise a warning.
    not_found = MAP2PATH['imgstore_name'].apply(
            lambda x: True if len(x)==0 else False).values
    if np.sum(not_found)>0:
        not_found = MAP2PATH.loc[not_found,:]
        for i,row in not_found.iterrows():
            warnings.warn('\n\nNo video found for day '
                          +'{}, run {}, camera {}.\n\n'.format(
                                  raw_day_dir.stem,*row.values)
                          )

    # keep only short imgstore_name (experiment_day_dir/imgstore_name_dir)
    MAP2PATH['imgstore_name'] = MAP2PATH['imgstore_name'].apply(
            lambda x: [join(*ix.parts[-3:-1]) for ix in x])

    # merge dataframes to store imgstore_name for each metadata row
    out_metadata = pd.merge(
            metadata,
            MAP2PATH[['imaging_run_number','camera_serial','imgstore_name']],
            how='outer',on=['imaging_run_number','camera_serial'])

    # explode metadata to have one row per video in case of multiple videos
    # with the same data,run and camera number
    out_metadata = explode_df(out_metadata, ['imgstore_name'])

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

def get_day_metadata(
        imaging_plate_metadata, manual_metadata_file,
        merge_on=['imaging_plate_id'], n_wells=96, saveto=None,
        del_if_exists=False, include_imgstore_name = True, raw_day_dir=None
        ):
    """
    @author: em812
    Incorporates the robot metadata and the manual metadata of the hydra rigs
    to get all the metadata for a given day of experiments.
    Also, it adds the imgstore_name of the hydra vidoes.

    param:
        imaging_plate_metadata: pandas dataframe
                Dataframe containing the metadata for all the
                wells of each imaging plate (If the robot was used, then this
                is the robot_metadata obtained by the
                robot_to_metadata.merge_robot_metadata function)
        manual_metadata: .csv file path
                File with the details of rigs and runs
                (see README for minimum required fields)
        merge_on: column name or list of columns
                Column(s) in common in imaging_plate_metadata and
                manual_metadata, that can be used to merge them
        n_wells: integer
                number of wells in imaging plate
        saveto: .csv file path
                File path to save the day metadata in (if None, the
                metadata are saved in the same folder as the manual metadata)
        del_if_exists: boolean
                If True and the metadata file exists in the defined
                path, the existing file will be deleted and a new file will be
                created. If False, the operation will be aborted and a warning
                will be produced.
        include_imgstore_name: boolean
                If True, the function will call the add_imgstore_name function
                to look in the raw_day_dir and get the imgstore names for each
                row in the metadata.
        raw_day_dir: directory path
                Path of the directory containing the RawVideos for the
                specific day of experiments. It is used to get the imgstore
                names. If None, then the standard file structure is assumed,
                and the path is obtained by the path of the auxiliary files
                for the specific day, by replacing AuxiliaryFile by RawVideos.

    return:
        metadata: pandas dataframe
            dataframe with day metadata
    """
    #find the date of the hydra experiments
    date_of_runs = get_date_of_runs_from_aux_files(manual_metadata_file)

    aux_day_dir = manual_metadata_file.parent
    if saveto is None:
        saveto = aux_day_dir / '{}_day_metadata.csv'.format(date_of_runs)

    if saveto.exists():
        if  del_if_exists:
            warnings.warn('\n\nMetadata file {} already exists.'.format(saveto)
                          +' File will be overwritten.\n\n')
            saveto.unlink()
        else:
            warnings.warn('\n\nMetadata file {} already exists.'.format(saveto)
                          +' Nothing to do here.\n\n')
            return

    manual_metadata = pd.read_csv(manual_metadata_file, index_col=False)

    if 'date_yyyymmdd' not in manual_metadata.columns:
        manual_metadata['date_yyyymmdd'] = str(date_of_runs)


    # make sure there is overlap in the image_plate_id
    if not np.all(
            np.isin(imaging_plate_metadata[merge_on],manual_metadata[merge_on])
            ):
        warnings.warn('There are {} values in the imaging '.format(merge_on)
                    +'plate metadata that do not exist in the manual '
                    +'metadata. These plates will be dropped from the day'
                    +'metadata file.')

    # merge two dataframes
    metadata = pd.merge(
            imaging_plate_metadata, manual_metadata, how='inner',
            left_on=merge_on,right_on=merge_on,indicator=False
            )

    # clean up and rearrange
    metadata.rename(columns={'destination_well':'well_name'}, inplace=True)

    # add imgstore name
    if include_imgstore_name:
        if raw_day_dir is None:
            raw_day_dir = Path(
                    str(aux_day_dir).replace('AuxiliaryFiles','RawVideos')
                    )
        if raw_day_dir.exists:
            metadata = add_imgstore_name(
                    metadata,raw_day_dir,n_wells=n_wells
                    )
            #check_dates_in_yaml(metadata,raw_day_dir)
        else:
            warnings.warn("\nRawVideos day directory was not found. "
                          +"Imgstore names cannot be added to the metadata.\n",
                          +"Path {} not found.".format(raw_day_dir)
                          )

    # Save to csv file
    print('Saving metadata file: {} '.format(saveto))
    metadata.to_csv(saveto, index=False)

    return metadata


#%%
# STEP 3:
def concatenate_days_metadata(
        aux_root_dir, list_days=None, saveto=None
        ):
    """
    @author: em812
    Reads all the yyyymmdd_day_metadata.csv files from the different days of
    experiments and creates a full metadata file all_metadata.csv.

    param:
        aux_root_dir: path to directory
            Root AuxiliaryFiles directory containing all the folders
            for the individual days of experiments
        list_days: list like object
            List of folder names (experiment days) to read from.
            If None (default), it reads all the subfolders in the root_dir.
        saveto: path to .csv file
            Filename where all compiled metadata will be saved

    return:
        None
    """
    aux_root_dir = Path(aux_root_dir)

    if saveto is None:
        saveto = aux_root_dir.joinpath('metadata.csv')

    if list_days is None:
        list_days = [d for d in aux_root_dir.glob("*") if d.is_dir()]


    ## Compile all metadata from different days
    meta_files = []
    for day in list_days:
        ## Find the correct day_metadata file
        auxfiles = [file for file in aux_root_dir.joinpath(day).glob('*_day_metadata.csv')]

        # checks
        if len(auxfiles) > 1:
            is_ok = np.zeros(len(auxfiles)).dtype(bool)
            for ifl,file in enumerate(auxfiles):
                if len(file.stem.replace('_day_metadata',''))==8:
                    is_ok[ifl]=True
            if np.sum(is_ok)==1:
                auxfiles = [file
                            for ifl,file in enumerate(auxfiles)
                            if is_ok[ifl]]
            else:
                raise ValueError('More than one *_day_metatada.csv files found'
                                 +'in {}. Compilation '.format()
                                 +'of metadata is aborted.')

        meta_files.append(auxfiles[0])

    all_meta = []
    for file in meta_files:
        all_meta.append(pd.read_csv(file))

    all_meta = pd.concat(all_meta,axis=0)
    all_meta.to_csv(saveto,index=False)

    return

#%%
if __name__ == '__main__':
    # Example 1:
    # Input
    day_root_dir = Path('/Users/em812/Data/Hydra_pilot/AuxiliaryFiles/'
                        +'20191108_tierpsytools_dev')
    raw_day_dir = Path('/Volumes/behavgenom$/Ida/Data/Hydra/PilotDrugExps/'
                        +'RawVideos/20191108')
    sourceplate_file = day_root_dir / '20191107_sourceplates.csv'
    manual_meta_file = day_root_dir / '20191108_manual_metadata.csv'

    # Save to
    #robot_metadata_file = day_root_dir.joinpath('20191107_robot_metadata.csv')
    metadata_file = day_root_dir.joinpath('20191108_day_metadata.csv')

    # Run
    robot_metadata = merge_robot_metadata(sourceplate_file, saveto=False)
    day_metadata = get_day_metadata(robot_metadata, manual_meta_file,
                                    saveto=metadata_file,del_if_exists=True,
                                    raw_day_dir=raw_day_dir)

    # Example 2:
    # Input
    aux_root_dir = Path(
            '/Volumes/behavgenom$/Ida/Data/Hydra/PilotDrugExps/AuxiliaryFiles')
    day_root_dirs = [d for d in aux_root_dir.glob("*") if d.is_dir()]

    sourceplate_files = [
            [file for file in d.glob('*_sourceplates.csv')]
            for d in day_root_dirs
            ]
    manual_meta_files = [
            [file for file in d.glob('*_manual_metadata.csv')]
            for d in day_root_dirs
            ]

    # Saveto
    metadata_files = [d.joinpath('{}_day_metadata.csv'.format(d.stem))
            for d in day_root_dirs]

    # Run
    for day,source,manual_meta,saveto in zip(
            day_root_dirs,sourceplate_files,manual_meta_files,metadata_files):
        if len(source)!=1:
            print('There is not a unique sourceplates file in '
                  +'day {}. Metadata cannot be compiled'.format(day))
            continue
        if len(manual_meta)!=1:
            print('There is not a unique manual_metadata file in '
                  +'day {}. Metadata cannot be compiled'.format(day))
            continue
        robot_metadata = merge_robot_metadata(source[0], saveto=False)
        day_metadata = get_day_metadata(
                robot_metadata, manual_meta[0], saveto=saveto[0])

    #%%Example 3:
    day_root_dir = Path('/Volumes/behavgenom$/Ida/Data/Hydra/ICDbacteria/AuxiliaryFiles/20191122')
    sourceplate_file = day_root_dir / '20191122_sourceplates.csv'
    manual_meta_file = day_root_dir / '20191122_manual_metadata.csv'
    metadata_file = day_root_dir.joinpath('20191122_day_metadata.csv')

    plate_metadata = populate_96WPs(sourceplate_file,
                                  entire_rows=True,
                                  saveto= None,
                                  del_if_exists = False)
    day_metadata = get_day_metadata(plate_metadata, manual_meta_file, saveto=metadata_file)

    #%% Example 4
    day_root_dir = Path('/Volumes/behavgenom$/Ida/Data/Hydra/SyngentaScreen/AuxiliaryFiles/20191205')
    sourceplate_file = day_root_dir / '20191204_sourceplates.csv'
    wormsorter_file = day_root_dir / '20191205_wormsorter.csv'
    manual_meta_file = day_root_dir / '20191205_manual_metadata.csv'
    metadata_file = day_root_dir / '20191205_day_metadata.csv'
    bad_wells_file = day_root_dir / '20191204_robot_bad_imaging_wells.csv'

    plate_metadata = populate_96WPs(wormsorter_file)
    robot_metadata = merge_robot_metadata(sourceplate_file)
    concat_metadata = merge_robot_wormsorter(day_root_dir,
                                             robot_metadata,
                                             plate_metadata,
                                             bad_wells_file)
    day_metadata = get_day_metadata(concat_metadata,
                                    manual_meta_file,
                                    saveto=metadata_file,
                                    del_if_exists=True)