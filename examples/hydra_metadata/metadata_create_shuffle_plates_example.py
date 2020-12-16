#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 23 April 2020

@author: ibarlow

Example script to generate shuffled source plates from a library plate

Sourceplates file required for each robot run with these headings.
    (additional headings allowed but these are *required*):

    - drug_type (or whatever is in the well)
    - source_well
    - column
    - library_plate_number
    - source_plate_number
    - source_plate_id
    - source_robotslot
    - robot_run_number
    - robot_runlog_filename

Robot runlogs refecerence in 'robot_runlog_filename' required in path

Preprocessing of runlogs is only required in order to remove the rows from
the robot runlog when the robot was doing repeated pipetting
up and down to mix the drug. Only needs to be done one time.

"""
import pandas as pd
from pathlib import Path
from tierpsytools.hydra.compile_metadata import merge_robot_metadata
from tierpsytools import EXAMPLES_DIR
import warnings

AUXILIARY_FILES = Path(EXAMPLES_DIR) / 'hydra_metadata' / 'data' / \
    'AuxiliaryFiles' / 'sourceplates'
PREPROCESSING_REQUIRED = False
OVERWRITE = True

sourceplate_files = list(
            AUXILIARY_FILES.rglob('2020SygentaLibrary3doses*'))
sourceplate_files =[i for i in sourceplate_files if 'shuffled' not in str(i)]

# %%
def preprocessing_robot_runlog(runlog_file):
    """Function to preprocess the runlogs and remove all the mixing pipetting
    steps
    Input
        runlog_file: .csv of the robot runlog (converted from .txt using
                                               Luigi's function)
    Output
        runlog_file_clean: .csv of the cleaned up runlog
        """
    rlog = pd.read_csv(file)
    rlog = rlog.drop(rlog[rlog['source_slot'] == rlog['destination_slot']
                          ].index)

    outfile = runlog_file.parent / (runlog_file.stem + '_clean.csv')
    rlog.to_csv(outfile, index=False)

    return rlog

# %%
if __name__ == '__main__':

    if PREPROCESSING_REQUIRED:
        robot_logs = list(AUXILIARY_FILES.rglob('*runlog.csv'))

        for file in robot_logs:
            preprocessing_robot_runlog(file)

        # update the sourceplates file
        for file in sourceplate_files:
            warnings.warn('{} file being edited to update robot log'.format(
                file))
            splate = pd.read_csv(file)
            splate.loc[:, 'robot_runlog_filename'] = [r[
                                            'robot_runlog_filename'].replace(
                                            'runlog.csv', 'runlog_clean.csv')
                                            for i, r in splate.iterrows()
                                                ]
            splate.to_csv(file, index=False)

    # now do the mapping
    for file in sourceplate_files:
        robot_metadata = merge_robot_metadata(file,
                                              saveto=None,
                                              del_if_exists=True,
                                              compact_drug_plate=True,
                                              drug_by_column=False)
        robot_metadata.sort_values(by=['source_plate_number',
                                       'destination_well'],
                                   ignore_index=True,
                                   inplace=True)

        # create shuffle plate id
        robot_metadata['shuffled_plate_id'] = [r.source_plate_id +
                                               '_sh%02d' %(r.robot_run_number)
                                               for i, r in
                                               robot_metadata.iterrows()]
        robot_metadata['is_bad_well'].fillna(False,
                                             inplace=True)

        outfile = Path(str(file).replace('.csv', '_shuffled.csv'))
        if outfile.exists():
            if OVERWRITE == False:
                warnings.warn('shuffled sourceplate file already exists for: {}\n'.
                              format(file) +
                              'File not overwritten')
            else:
                warnings.warn('shuffled sourceplate file overwritten :\n {}'.
                              format(file))

                robot_metadata.to_csv(outfile,
                                      index=False)
                del robot_metadata
