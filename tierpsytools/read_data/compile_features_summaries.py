#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions that can be used to compile indivudual
features_summaries and filenames_summaries (assuming tierpsy format) and
create one features_summaries and one filenames_summaries file containing all
the results.

Created on Mon Apr  6 12:29:50 2020

@author: em812
"""
from pathlib import Path
import pandas as pd
import warnings
from time import time


def find_fname_summaries(feat_files):
    fname_files = []
    for feat_fl in feat_files:
        fname_fl = None
        # 1) look for filenames_summaries filename in features_summaries file headers
        with open(feat_fl, 'r') as fid:
            comment_line = True
            while comment_line:
                line = fid.readline()
                if 'FILENAMES SUMMARY FILE' in line:
                    fname_fl = Path(line.split(',')[1].replace('\n',''))
                    break
                if line[0] != '#':
                    comment_line = False
                    warnings.warn(
                        'filenames_summaries filename not found in the ' +
                        'features_summaries file. Looking for the file in ' +
                        'features_summaries path...')

        # 2) If not there, look for filenames_summaries in the same folder where
        # features_summaries is located
        if fname_fl is None:
            if (feat_fl.parent / (feat_fl.stem.replace('features','filenames')+'.csv')).exists():
                fname_fl = (feat_fl.parent / (feat_fl.stem.replace('features','filenames')+'.csv'))
                print('filenames_summaries file found.')
            else:
                warnings.warn(
                    'No filenames_summaries file found matching the ' +
                    'features_summaries file {}.'.format(feat_fl))

        # If the filenames summary file is not found, None will be appended to
        # the fname_files list so that the one-to-one correspondance with the
        # the feat_files list is maintained.

        fname_files.append(fname_fl)

    return fname_files

def check_summary_type(feat_files, fname_files):
    sum_type = []
    for file in feat_files + fname_files:
        tmp = ''.join(file.stem.split('_')[2:])
        tmp = ''.join([l for l in tmp if not l.isdigit()])
        sum_type.append(tmp)
    if not all(stype == sum_type[0] for stype in sum_type):
        raise ValueError('The summary files specified are not of the same type.')
    return

def _get_fname_files(feat_files):
    fname_files = []
    for file in feat_files:
        with open(file, "r") as fid:
            header = fid.readline()
            if not header.startswith('#'):
                raise ValueError(
                    'No header found in features summaries file. ' +
                    'Corresponding filename summaries file cannot be retrieved.')
            fname_files.append(header.replace('#','').replace(' ', ''))
    assert len(feat_files) == len(fname_files)
    return fname_files

def _copy_commented_headers(source_file, dest_file, comment='#'):
    fid = open(source_file, 'r')
    fid_comp = open(dest_file, 'w')

    for i in range(20):
        line = fid.readline()
        if line[0] == '#':
            fid_comp.write(line)
        else:
            break
    fid.close()
    fid_comp.close()
    return

def _compile_columns(file_list):
    from pandas.errors import EmptyDataError

    # Read one line from each dataframe
    dfsamples = []
    for file in file_list:
        try:
            dfsamples.append(pd.read_csv(
                file, index_col=None, header=0, nrows=1, comment='#'))
        except EmptyDataError:
            raise EmptyDataError('Summary file empty: \n{}'.format(file))

    # Compare number of columns
    ncols = [df.shape[1] for df in dfsamples]
    if not all([n==ncols[0] for n in ncols]):
        warnings.warn('The dataframes to compile do not have the same number of columns.')

    # Compile all columns
    columns = pd.concat(dfsamples, axis=0, sort=False).columns.to_list()

    return columns

def _write_columns_header(columns, filename):
    with open(filename,'a') as fid:
        fid.write(','.join(columns)+'\n')
    return

def _sort_columns(df, columns):
    from numpy import nan
    missing = [col for col in columns if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = nan
    return df[columns]

def compile_tierpsy_summaries(
        feat_files, compiled_feat_file, compiled_fname_file, fname_files=None):
    """
    Reads feature summaries from different days of the same experiment and
    compiled them to a single feature summaries file and a single filenames
    summaries file.

    Parameters
    ----------
    feat_files : list
        The list of feature summaries files to compile.
    compiled_feat_file : path
        The full path to the compiled features summaries file.
    compiled_fname_file : path
        The full path to the compiled filenames summaries file.
    fname_files : list (optional)
        The list of filenames summaries files that correspond to the feat_files
        list. If None, the filenames summaries names will be read from the
        header of the feat_files.
    Raises
    ------
    ValueError
        If the compiled_feat_file or the compiled_fname_file already exist.

    Returns
    -------
    None.

    """
    import pdb

    if compiled_feat_file.exists() or compiled_fname_file.exists():
        raise ValueError(
            'Files with the same name already exists in the ' +
            'specified location. To compile summaries in the specified file, '+
            'remove or rename the existing files.'
            )

    if fname_files is None:
        fname_files = _get_fname_files(feat_files)

    check_summary_type(feat_files, fname_files)

    _copy_commented_headers(feat_files[0], compiled_feat_file)
    _copy_commented_headers(fname_files[0], compiled_fname_file)

    ## Get all the columns from the different files:
    feat_columns = _compile_columns(feat_files)
    fname_columns = _compile_columns(fname_files)

    _write_columns_header(feat_columns, compiled_feat_file)
    _write_columns_header(fname_columns, compiled_fname_file)

    # Append dataframes to files
    counter = 0
    for i, (feat_fl, fname_fl) in enumerate(zip(feat_files, fname_files)):
        print('Appending dataframes to compiled files {}/{}...'.format(i+1, len(feat_files)))
        print(feat_fl)
        print(fname_fl)
        st_time = time()
        fnames = pd.read_csv(fname_fl, header=0, comment='#')
        nfiles = fnames['file_id'].max() + 1
        fnames['file_id'] = fnames['file_id'] + counter
        fnames = _sort_columns(fnames, fname_columns)
        with open(compiled_fname_file, 'a') as fid:
            fnames.to_csv(fid, header=False, index=False)

        try:
            feat = pd.read_csv(feat_fl, header=0, comment='#')
        except:
            pdb.set_trace()
        feat['file_id'] = feat['file_id'] + counter
        feat = _sort_columns(feat, feat_columns)
        with open(compiled_feat_file, 'a') as fid:
            feat.to_csv(fid, header=False, index=False)
        print('Done in {:.2f} sec.'.format(time()-st_time))

        counter += nfiles

    return

if __name__ == "__main__":

    ## EXAMPLE OF USE
    # Assume we have three days of experiments and we have obtained
    # feature and filenames summaries with tierpsy for each day individually.
    # Sample data used here can be found in the following folder of the
    # tierpsytools package:
    # examples/read_data/compile_features_summaries

    from tierpsytools import EXAMPLES_DIR

    root_dir = Path(EXAMPLES_DIR) / 'read_data' / 'compile_features_summaries'
    compiled_feat_file = Path(root_dir) / 'features_summaries_compiled.csv'
    compiled_fname_file = Path(root_dir) / 'filenames_summaries_compiled.csv'

    if compiled_feat_file.exists():
        compiled_feat_file.unlink()
    if compiled_fname_file.exists():
        compiled_fname_file.unlink()

    ## Get a list of the features_summaries files that you want to compile
    feat_files = [file for file in Path(root_dir).rglob('features*.csv')]

    ## Find the matching filenames_summaries
    fname_files = find_fname_summaries(feat_files)

    # keep only the features files for which you found matching filenames_summaries
    feat_files = [feat_fl for feat_fl,fname_fl in zip(feat_files, fname_files)
                  if fname_fl is not None]
    fname_files = [fname_fl for fname_fl in fname_files
                   if fname_fl is not None]

    ## Compile the summaries files in the specified files
    compile_tierpsy_summaries(feat_files, fname_files, compiled_feat_file, compiled_fname_file)



