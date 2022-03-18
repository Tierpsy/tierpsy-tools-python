#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:25:41 2019

@author: em812
"""
import pandas as pd
from pathlib import Path
from tierpsytools.hydra.hydra_helper import exract_randomized_by
from tierpsytools.hydra.hydra_helper import rename_out_meta_cols, explode_df, column_from_well, row_from_well
import re
import warnings
import numpy as np

#%% General functions for all types of experiments
def populate_6WPs(worm_sorter, saveto=None, del_if_exists=False):
    """
    populate_6WPs Wrapper for populate_MWPs(n_columns=3, n_rows=2, **kwargs)
    Function to explode summary info in wormsorter files and create
    plate_metadata with all the information per well per unique imaging plate.
    Works with plates that have been filled row-wise and column-wise
    consecutively

    Parameters
    ----------
    worm_sorter : .csv file with headers 'start_well', 'end_well' and details
        of strains and media in range of wells.
    saveto : None, 'default' or path to csv file
        If None, the plate metadata will not be saved in a file.
        If 'default', the plate_metadata will be saved in a csv at the same
        location as the wormsorter file.
        If path, the plate_metadata will be saved in a csv at the path location.
        The default is None.
    del_if_exists : Bool, optional
        If a file is found at the location specified in saveto and del_if_exists
        is True, then the file will be deleted and regenerated. If del_if_exists
        is False, then the file will not be deleted and the function will exit
        with error. The default is False.

    Returns
    -------
    plate_metadata: one line per well; can be used in get_day_metadata
            function to compile with manual metadata
    """
    plate_metadata = populate_MWPs(
        worm_sorter,
        n_rows=2, n_columns=3,
        saveto=saveto, del_if_exists=del_if_exists
        )
    return plate_metadata


def populate_24WPs(worm_sorter, saveto=None, del_if_exists=False):
    """
    populate_24WPs Wrapper for populate_MWPs(n_columns=6, n_rows=4, **kwargs)
    Function to explode summary info in wormsorter files and create
    plate_metadata with all the information per well per unique imaging plate.
    Works with plates that have been filled row-wise and column-wise
    consecutively

    Parameters
    ----------
    worm_sorter : .csv file with headers 'start_well', 'end_well' and details
        of strains and media in range of wells.
    saveto : None, 'default' or path to csv file
        If None, the plate metadata will not be saved in a file.
        If 'default', the plate_metadata will be saved in a csv at the same
        location as the wormsorter file.
        If path, the plate_metadata will be saved in a csv at the path location.
        The default is None.
    del_if_exists : Bool, optional
        If a file is found at the location specified in saveto and del_if_exists
        is True, then the file will be deleted and regenerated. If del_if_exists
        is False, then the file will not be deleted and the function will exit
        with error. The default is False.

    Returns
    -------
    plate_metadata: one line per well; can be used in get_day_metadata
            function to compile with manual metadata
    """

    plate_metadata = populate_MWPs(
        worm_sorter,
        n_rows=4, n_columns=6,
        saveto=saveto, del_if_exists=del_if_exists
        )
    return plate_metadata


def populate_96WPs(worm_sorter, saveto=None, del_if_exists=False, **kwargs):
    """
    populate_96WPs Wrapper for populate_MWPs(n_columns=12, n_rows=8, **kwargs)
    Function to explode summary info in wormsorter files and create
    plate_metadata with all the information per well per unique imaging plate.
    Works with plates that have been filled row-wise and column-wise
    consecutively

    Parameters
    ----------
    worm_sorter : .csv file with headers 'start_well', 'end_well' and details
        of strains and media in range of wells.
    saveto : None, 'default' or path to csv file
        If None, the plate metadata will not be saved in a file.
        If 'default', the plate_metadata will be saved in a csv at the same
        location as the wormsorter file.
        If path, the plate_metadata will be saved in a csv at the path location.
        The default is None.
    del_if_exists : Bool, optional
        If a file is found at the location specified in saveto and del_if_exists
        is True, then the file will be deleted and regenerated. If del_if_exists
        is False, then the file will not be deleted and the function will exit
        with error. The default is False.

    Returns
    -------
    plate_metadata: one line per well; can be used in get_day_metadata
            function to compile with manual metadata
    """
    for k in ['n_columns', 'n_rows']:
        if k in kwargs:
            warnings.warn(
                f'Passing {k} to populate_96WPs is deprecated.'
                '`populate_96WPs(...)` is a wrapper for '
                '`populate_MWPs(..., n_columns=12, n_rows=8)`. '
                'A future update will ignore n_columns and n_rows '
                'passed to this function. '
                'Please update your code'
            )
    n_rows = kwargs['n_rows'] if 'n_rows' in kwargs else 8
    n_columns = kwargs['n_columns'] if 'n_columns' in kwargs else 12

    plate_metadata = populate_MWPs(
        worm_sorter,
        n_rows=8, n_columns=12,
        saveto=saveto, del_if_exists=del_if_exists
        )
    return plate_metadata


def populate_MWPs(
        worm_sorter, n_columns=12, n_rows=8, saveto=None, del_if_exists=False):
    """
    @author: ilbarlow

    Function to explode summary info in wormsorter files and create plate_metadata
    with all the information per well per unique imaging plate.
    Works with plates that have been filled row-wise and column-wise
    consecutively

    Parameters
    ----------
    worm_sorter : .csv file with headers 'start_well', 'end_well' and details
        of strains and media in range of wells.
    saveto : None, 'default' or path to csv file
        If None, the plate metadata will not be saved in a file.
        If 'default', the plate_metadata will be saved in a csv at the same
        location as the wormsorter file.
        If path, the plate_metadata will be saved in a csv at the path location.
        The default is None.
    del_if_exists : Bool, optional
        If a file is found at the location specified in saveto and del_if_exists
        is True, then the file will be deleted and regenerated. If del_if_exists
        is False, then the file will not be deleted and the function will exit
        with error. The default is False.

    Returns
    -------
    plate_metadata: one line per well; can be used in get_day_metadata
            function to compile with manual metadata
    """
    import string
    worm_sorter = Path(worm_sorter)

    DATE = worm_sorter.stem.split('_')[0]
    wormsorter_log_fname = worm_sorter.parent / '{}_wormsorter_errorlog.txt'.format(DATE)
    wormsorter_log_fname.touch()

    # parameters for the plates
    column_names = np.arange(1, n_columns+1)
    row_names = list(string.ascii_uppercase[0:n_rows])
    well_names = [i+str(b) for i in row_names for b in column_names]
    total_nwells = len(well_names)

    print('Total number of wells: {}'.format(total_nwells))

    # checking if and where to save the file
    if saveto is not None:
        if saveto == 'default':
            saveto = Path(worm_sorter).parent / ('{}_plate_metadata.csv'.format(DATE))

        if saveto.exists():
            if del_if_exists:
                warnings.warn('\nPlate metadata file {} already '.format(saveto)
                              + 'exists. File will be overwritten.')
                saveto.unlink()
            else:
                raise Exception('\nPlate metadata file {} already '.format(saveto)
                              + 'exists. Nothing to do here. If you want to '
                              + 'recompile the wormsorter metadata, rename or delete the '
                              + 'exisiting file or del_if_exists to True.')
        else:
            print ('saving to {}'.format(saveto))

    # import worm_sorter metadata
    worm_sorter_df = pd.read_csv(worm_sorter)

    # find the start and end rows and columns
    worm_sorter_df = worm_sorter_df.dropna(axis=0, how='all')
    worm_sorter_df['start_row'] = row_from_well(worm_sorter_df.start_well)
    worm_sorter_df['end_row'] = row_from_well(worm_sorter_df.end_well)
    worm_sorter_df['row_range'] = [[chr(c) for c in np.arange(ord(r.start_row),
                                                              ord(r.end_row)+1)]
                                   for i, r in worm_sorter_df.iterrows()]
    worm_sorter_df['start_column'] = column_from_well(worm_sorter_df.start_well)
    worm_sorter_df['end_column'] = column_from_well(worm_sorter_df.end_well)
    worm_sorter_df['column_range'] = [list(np.arange(r.start_column,
                                                 r.end_column+1))
                                      for i, r in worm_sorter_df.iterrows()]

    worm_sorter_df['well_name'] = [[i+str(b) for i in r.row_range
                                    for b in r.column_range]
                                for i, r in worm_sorter_df.iterrows()]

    plate_metadata = explode_df(worm_sorter_df, 'well_name')

    # do check to make sure there aren't multiple  plates
    plate_errors =  []
    unique_plates = plate_metadata['imaging_plate_id'].unique()
    for plate in unique_plates:
        if plate_metadata[plate_metadata['imaging_plate_id'] == plate].shape[0] > total_nwells:
            warnings.warn('{}: more than {} wells'.format(plate, total_nwells))
            plate_errors.append(plate)
            with open(wormsorter_log_fname, 'a') as fid:
                fid.write(plate + '\n')
    if len(plate_errors) == 0:
        with open(wormsorter_log_fname, 'a') as fid:
            fid.write ('No wormsorter plate errors \n')

    if saveto is not None:
        plate_metadata.to_csv(saveto, index=False)

    return plate_metadata


def get_day_metadata(
        complete_plate_metadata, hydra_metadata_file,
        merge_on=['imaging_plate_id'], n_wells=96,
        run_number_regex=r'run\d+_',
        saveto=None,
        del_if_exists=False, include_imgstore_name = True, raw_day_dir=None
        ):
    """
    @author: em812
    Incorporates the robot metadata and the manual metadata of the hydra rigs
    to get all the metadata for a given day of experiments.
    Also, it adds the imgstore_name of the hydra vidoes.

    param:
        complete_plate_metadata: pandas dataframe
                Dataframe containing the metadata for all the
                wells of each imaging plate (If the robot was used, then this
                is the complete_plate_metadata obtained by the
                merge_robot_wormsorter function)
        hydra_metadata_file: .csv file path
                File with the details of rigs and runs
                (see README for minimum required fields)
        merge_on: column name or list of columns
                Column(s) in common in complete_plate_metadata and
                hydra_metadata, that can be used to merge them
        run_number_regex: regex defining the format of the run number in
                in the raw video file names. Default translates to 'run#_'.
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
    from tierpsytools.hydra.hydra_helper import \
        get_date_of_runs_from_aux_files, add_imgstore_name

    hydra_metadata_file = Path(hydra_metadata_file)

    #find the date of the hydra experiments
    date_of_runs = get_date_of_runs_from_aux_files(hydra_metadata_file)

    aux_day_dir = hydra_metadata_file.parent
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

    hydra_metadata = pd.read_csv(hydra_metadata_file, index_col=False)

    if 'date_yyyymmdd' not in hydra_metadata.columns:
        hydra_metadata['date_yyyymmdd'] = str(date_of_runs)


    # make sure there is overlap in the image_plate_id
    if not np.all(
            np.isin(complete_plate_metadata[merge_on],hydra_metadata[merge_on])
            ):
        warnings.warn('There are {} values in the imaging '.format(merge_on)
                    +'plate metadata that do not exist in the manual '
                    +'metadata. These plates will be dropped from the day'
                    +'metadata file.')

    # merge two dataframes
    metadata = pd.merge(
            complete_plate_metadata, hydra_metadata, how='inner',
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
                    metadata, raw_day_dir, n_wells=n_wells,
                    run_number_regex=run_number_regex
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
        all_meta: compiled metadata dataframe
    """

    date_regex = r"\d{8}"
    aux_root_dir = Path(aux_root_dir)

    if saveto is None:
        saveto = aux_root_dir.joinpath('metadata.csv')

    if list_days is None:
        list_days = [d for d in aux_root_dir.glob("*") if d.is_dir()
                    and re.search(date_regex, str(d)) is not None]

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

    return all_meta


#%% Experiments without plate shuffling
def get_source_metadata(sourceplates_file, imaging2source_file):
    """
    author: @em812
    This function creates a dataframe with the drug (or other) content of every
    well of every imaging plate based on the corresponding source plate id and the
    information about the content of the source plates found in the
    sourceplates_file.

    Parameters
    ----------
    sourceplates_file : path
        The sourceplates file that defines the drug content of each unique
        source plate used in the given tracking day.
    imaging2source_file : path
        path to imaging2source_file which contain a simple mapping between
        every imaging plate screened in the given tracking day and the
        source plate used to make the imaging plate.

    Returns
    -------
    source_metadata : pandas dataframe
        A dataframe that contains information about the drug content of each
        well of every imaging plate screened in a given tracking day.

    """
    sourceplates = pd.read_csv(sourceplates_file)
    imag2source = pd.read_csv(imaging2source_file)

    source_metadata = pd.merge(
        sourceplates, imag2source, on='source_plate_id', how='outer')

    source_metadata = source_metadata.sort_values(by=['imaging_plate_id', 'well_name'])

    return source_metadata

def merge_basic_and_source_meta(
        plate_metadata, source_metadata,
        merge_on=['imaging_plate_id', 'well_name'],
        saveto=None, del_if_exists=False
        ):
    """
    Add the source_metadata to the plate_metadata to get the complete_plate_metadata
    (all the information about every well in every imaging plate at a given
     tracking day).

    Parameters
    ----------
    plate_metadata : pandas dataframe
        The output of populate_96WP.
    source_metadata : pandas dataframe
        data about the drug content of every well of every imaging_plate screened
        at a given day.
    merge_on : list of columns names, optional
        column names that can be found both in plate_metadata and source_metadata
        and which can be used to merge the two dataframes.
        The default is ['imaging_plate_id', 'well_name'].
    saveto : None or path to csv file, optional
        If not None, then the complete_plate_metadata will be saved in the
        csv file defined in the path. The default is None.
    del_if_exists : bool, optional
        Definesthe behaviour if saveto is not None and the file defined in saveto
        exists. If del_if_exists is True the file will ne deleted and overwriten.
        If del_if_exists in False the, the function will exit with error.
        The default is False.

    Returns
    -------
    complete_plate_metadata : pandas dataframe
        the complete plate metadata with all the infromation about every well
        of every imaging plate in a given tracking day.

    """

    # check if file exists
    if (saveto is not None) and (Path(saveto).exists()):
        if del_if_exists:
            warnings.warn('Plate metadata file {} already '.format(saveto)
                          + 'exists. File will be overwritten.')
            saveto.unlink()
        else:
            raise Exception('Merged plate and drug metadata file ' +
                          '{} already exists. Nothing to do here.'.format(saveto) +
                          'If you want to recompile the plate metadata,'
                          'rename or delete the exisiting file or set' +
                          'del_if_exists to True.')

    complete_plate_metadata = pd.merge(
        plate_metadata, source_metadata, on=merge_on,
        how='outer')

    if saveto is not None:
        complete_plate_metadata.to_csv(saveto, index=False)

    return complete_plate_metadata


#%% Experiments using OPENTRONS
def merge_robot_metadata(
        sourceplates_file, randomized_by='column', saveto=None,
        drug_by_column=True,
        compact_drug_plate=False,
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

        drug_by_column: Bool, if robot randomised by column and each column has the
        same drug at a single concentration (Eg Dec syngenta screen)

        compact_drug_plate: Bool, if each well is a different drug as a
        different concentration (eg Jan syngenta strain screen)

        saveto: path to .csv file
            The full path of the file where the robot metadata will be saved.
            If None, the robot metadata dataframe is not saved to disk.
        del_if_exists: boolean
            If True, then if the saveto file exists, it will be replaced.

    return:
        source_metadata: pandas dataframe
            Robot related metadata for the given day of experiments as dataframe

    """

    if saveto is None:
        date = sourceplates_file.stem.split('_')[0]
        saveto = Path(sourceplates_file).parent / (date+'_source_metadata.csv')

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
    sourceplates = sourceplates.dropna(axis=0, how='all', inplace=False)

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
    source_metadata = []
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
        if drug_by_column:
            out_meta = pd.merge(
                    source, robotlog, how='outer',
                    left_on=['source_plate_id', randomized_by],
                    right_on=['source_plate_id',randomized_by]
                    )

            # get unique imaging_plate_id
            out_meta['imaging_plate_id'] = out_meta[
                    ['source_plate_id','destination_slot']
                    ].apply(
                        lambda x: 'rr{:02}_sp{:02}_ds{:02}'.format(n_log+1, *x),
                        axis=1)

        elif compact_drug_plate:
            out_meta = pd.merge(
                    source, robotlog, how='outer',
                    left_on=['source_plate_id', 'source_well', randomized_by],
                    right_on=['source_plate_id', 'source_well', randomized_by],
                    )
            # remove wells where drug type is na (ie when half filled plate)
            out_meta = out_meta[out_meta['drug_type'].notna()]


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
        if drug_by_column:
            leading_cols = ['imaging_plate_id', 'source_plate_id',
                            'destination_slot']
        else:
            leading_cols = ['source_plate_id',
                            'destination_slot']
        end_cols = ['robot_runlog_id', 'robot_runlog_filename']

        out_meta = out_meta[
                leading_cols
                + list(out_meta.columns.difference(leading_cols + end_cols))
                + end_cols
                ]

        # append to list
        source_metadata.append(out_meta)

    # concatenate data from all the robot runlogs
    source_metadata = pd.concat(source_metadata, axis=0)

    if saveto is not False:
        source_metadata.to_csv(saveto, index=None)

    return source_metadata


def merge_robot_wormsorter(day_root_dir,
                           source_metadata,
                           plate_metadata,
                           bad_wells_csv=None, # check condition where no bad_wells_csv
                           merge_on=['imaging_plate_id', 'well_name'],
                           saveto=None,
                           del_if_exists=False,
                           ):
    """
    author: @ilbarlow
    Function for combining the outputs of the wormsorter and robot so that
    concise and comprehensive dataframe can be used in get_day_metadata
    Input:
    day_root_dir - root directory of metadata
    source_metadata - output from merge_robot_metadata function
    plate_metadata - output form populate_96WP function
    bad_wells_csv - .csv file listing imaging_plate_id and well_name
    Ouput:
    complete_plate_metadata - dataframe that can be used in get_day_metadata
    """
    from tierpsytools.hydra.hydra_helper import convert_bad_wells_lut

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

    source_metadata.rename(columns={'destination_well': 'well_name'},
                          inplace=True)

    # parse out bad wells in source_metadata into new column
    print('Finding any bad wells noted in the sourceplates and robot metadata')

    bad_source_lut = source_metadata[source_metadata['bad_wells'].notna()][
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


    # read in the bad imaging wells - noted by during robot/wormsorter runs
    if bad_wells_csv:
        robot_bad_wells_df = convert_bad_wells_lut(bad_wells_csv)
        bad_well_lut = pd.concat([robot_bad_wells_df,
                                  bad_well_lut], axis=0, sort=True
                                 ).reset_index(drop=True).drop_duplicates()
    else:
        print('no bad_wells.csv input to combine with robot metadata')

    # combine bad wells from both sources
    if not bad_well_lut.empty:
        print('Concatenating source_metadata, plate_metadata and bad wells')
        complete_plate_metadata = source_metadata.set_index(merge_on).join(
                                        [plate_metadata.set_index(merge_on),
                                         bad_well_lut.set_index(merge_on)],
                                     how='outer')
        complete_plate_metadata['is_bad_well'].fillna(False, inplace=True)

    else:
        print('No bad wells on this day of tracking; concatenating robot' +
              'and plate metadata')
        complete_plate_metadata = source_metadata.set_index(merge_on).join(
                plate_metadata.set_index(merge_on),
                how='outer')
        complete_plate_metadata['is_bad_well'] = False

    complete_plate_metadata.reset_index(drop=False,
                                inplace=True)
    # check that the number of output rows == input rows
    assert source_metadata.shape[0] == complete_plate_metadata.shape[0]

    complete_plate_metadata.to_csv(saveto, index=False)

    return complete_plate_metadata



# %% CHECKS
def day_metadata_check(day_metadata, day_root_dir, plate_size=96):

    """@author: ibarlow
    Function to check that the day_metadata is the correct size and has
    all the wells in triplicate
    Input:
        day_metadata - dataframe of the compiled metadata
        day_root_dir - directory to save output list to

    Output:
        list of files to check"""

    try:
        assert (day_metadata.shape[0] % plate_size == 0)

    except AssertionError:
        print ('Assertion Error - incorrect number of files')
        # if there number of rows of day metadata is not divisible by
        # 96 means there has been an issue with propagating
        files_to_check = []

        plate_list = list(day_metadata['imaging_plate_id'].unique())
        day_metadata_grouped = day_metadata.groupby('imaging_plate_id')
        for plate in plate_list:
            _checking = day_metadata_grouped.get_group(plate)
            if _checking.shape[0] % plate_size != 0:
                wells = _checking['well_name'].unique()
                print(plate, wells)
                for well in wells:
                    if (_checking['well_name'] == well).sum() != 3:
                        print(well)
                        files_to_check.append(
                                _checking[_checking['well_name'] == well]
                                ['imgstore_name'].to_list())
        if len(files_to_check)>0:
            files_to_check = [i for sublist in files_to_check for i in sublist]
            files_to_check = list(np.unique(files_to_check))
            print('These files need to be checked {}'.format(files_to_check))

            files_to_check = pd.Series(files_to_check)
            files_to_check.to_csv(day_root_dir / '{}_files_to_check.csv'.format(
                day_root_dir.stem))

            return files_to_check



def number_wells_per_plate(day_metadata, day_root_dir):
    """
    author @ibarlow

    Function that returns a csv listing all the wells and the total number of
    wells per plate
    Parameters
    ----------
    day_metadata : dataframe, must have columns 'imaging_plate_id' and 'well_name'
        DESCRIPTION.
    day_root_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    .csv and dataframe of plate summary

    """

    saveto = day_root_dir / '{}_wells_per_plate.csv'.format(day_root_dir.stem)

    imaging_plates = day_metadata['imaging_plate_id'].unique()

    plate_summary_df = []
    for plate in imaging_plates:
        plate_summary_df.append(pd.DataFrame().from_dict(
            {'imaging_plate_id': plate,
             'all_wells': [day_metadata[day_metadata[
                 'imaging_plate_id'] == plate]['well_name'].unique()],
             'number_wells': day_metadata[day_metadata[
                 'imaging_plate_id'] == plate]['well_name'].unique().shape[0]}))

    plate_summary_df = pd.concat(plate_summary_df)
    plate_summary_df.to_csv(saveto, index=False)

    return plate_summary_df




#%%
if __name__ == '__main__':
    pass
