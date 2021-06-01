#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:15:41 2020

@author: ibarlow

Functions for adding wells annotations from Luigi's gui to metadata

Requires:
    pytables=3.6.1
    numpy=1.17.5

"""

import pandas as pd
from pathlib import Path
import warnings
from tierpsytools.hydra.match_bluelight_videos import (
    match_bluelight_videos_in_folder)


def import_wells_annotations(annotations_file,
                             table_names=['/filenames_df',
                                          '/wells_annotations_df'],
                             imgstore_name='imgstore_prestim'):
    """
    Import tables from hdf5 files to combines them in a dataframe

    Parameters
    ----------
    annotations_file : .hdf5 file generated from the well annotator
        DESCRIPTION.
    table_names : names of the tables in annotations.hdf5 file
        DESCRIPTION. The default is ['/filenames_df',
        '/wells_annotations_df'].

    imgstore_name: default is imgstore_prestim

    Returns
    -------
    annotations : TYPE
        DESCRIPTION.

    """
    with pd.HDFStore(annotations_file) as fid:
        _fnames = fid[table_names[0]]
        _markings = fid[table_names[1]]

    # checks and warnings
    # first check for skipped videos: they are in the files df but not in the
    # markings
    skipped_files = _fnames.loc[
        ~_fnames['file_id'].isin(_markings['file_id']), 'filename'
        ].to_list()

    if len(skipped_files) > 0:
        warning_msg = (
            f'The following {len(skipped_files)} videos were not annotated:\n'
            + '\n'.join(skipped_files)
            + '\nThese videos will be ignored for the moment.'
            + '\nThis will be an error in the future.'
        )
        warnings.warn(warning_msg)

    # then check for skipped wells
    n_skipped_wells = _markings['well_label'].isin([0]).sum()
    if n_skipped_wells > 0:
        warnings.warn(
            f'{n_skipped_wells} wells were skipped and are not annotated.\n'
            'Treating them as bad wells.'
            )

    annotations_df = _fnames.merge(
        _markings[['file_id', 'well_label', 'well_name']],
        on='file_id',
        # right_index=True,
        validate='one_to_many')
    annotations_df[imgstore_name] = annotations_df.filename.apply(
                                        lambda x: x.split('/')[0])

    # collapsing the different values of well_label => is_bad_wells
    annotations_df['is_bad_well'] = annotations_df['well_label'] != 1

    return annotations_df


def import_wells_annotations_in_folder(
        aux_dir, search_string='*wells_annotations.hdf5', **kwargs):
    """
    Find annotations files in an AuxiliaryFiles directory and returns the
    annotations as a dataframe

    Parameters
    ----------
    aux_dir : Pathlib object
        DESCRIPTION.
    search_string : TYPE, optional
        DESCRIPTION. The default is '*wells_annotations.hdf5'.

    Returns
    -------
    annotations_df : TYPE
        DESCRIPTION.

    """
    # find all annotations files in aux_dir matching search_string
    annotations_files = list(Path(aux_dir).rglob(search_string))
    # loop, read each annotation as a dataframe, concatenate as a single df
    annotations_df = []
    for f in annotations_files:
        annotations_df.append(import_wells_annotations(f, **kwargs))
    annotations_df = pd.concat(annotations_df, axis=0, ignore_index=True)

    return annotations_df


def match_rawvids_annotations(rawvid_dir,
                              annotations_df,
                              bluelight_names=['imgstore_prestim',
                                               'imgstore_bluelight',
                                               'imgstore_poststim']):
    """
    Find raw video metadata.ymls, extract out the
    imgstore names and match annotations from prestim videos and
    propagates to the bluelight and poststim videos

    Parameters
    ----------
    rawvid_dir : pathlib Path
        Directory where to look for raw videos.
    annotations_df : output from import_wells_annotations
        DESCRIPTION.
    bluelight_names : TYPE, optional
        DESCRIPTION. The default is ['imgstore_prestim',
        'imgstore_bluelight',
        'imgstore_poststim'].

    Returns
    -------
    matched_long : TYPE
        DESCRIPTION.

    """
    matched_rawvids = match_bluelight_videos_in_folder(rawvid_dir)

    matched_raw_annotations = matched_rawvids.merge(annotations_df,
                                                    how='outer',
                                                    on='imgstore_prestim',
                                                    validate='one_to_many')

    matched_long = matched_raw_annotations.melt(id_vars=['well_name',
                                                         'well_label',
                                                         'is_bad_well'],
                                                value_vars=bluelight_names,
                                                value_name='imgstore')
    matched_long.drop(columns=['variable'],
                      inplace=True)
    # this is now done earlier:
    # matched_long['is_bad_well'] = matched_long['well_label'] != 1

    return matched_long


def update_metadata(aux_dir, matched_long, saveto=None, del_if_exists=False):
    """
    Concatenate the wells annotations with the metadata

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.
    matched_long : TYPE
        DESCRIPTION.
    saveto: filename to save output

    del_if_exists: Boolean to overwrite or not

    Returns
    -------
    update_metadata : TYPE
        DESCRIPTION.

    """
    warnings.warn(
        'This function is deprecated. '
        'Use `update_metadata_with_wells_annotations` instead',
        category=DeprecationWarning)

    metadata_fname = list(Path(aux_dir).rglob('metadata.csv'))

    if len(metadata_fname) > 1:
        warnings.warn(
            'More than one metadata file in this directory: \n' +
            f'{metadata_fname} \naborting.'
            )
        return None

    # saving and overwriting checks
    if saveto is None:
        saveto = aux_dir / 'wells_updated_metadata.csv'

    # check if file exists
    if (saveto is not False) and (saveto.exists()):
        if del_if_exists:
            warnings.warn(
                f'Wells annotations file, {saveto} already exists. '
                'File will be overwritten.')
            saveto.unlink()
        else:
            warnings.warn(
                f'Wells annotations file, {saveto} already exists. '
                'Nothing to do here. If you want to recompile the day metadata'
                ', rename or delete the existing file.')
            return None

    # read metadata
    metadata = pd.read_csv(metadata_fname[0])
    if metadata['imgstore_name'].isna().sum() > 0:
        print('Nan values in imgstore names')
        metadata = metadata[metadata['imgstore_name'].notna()]
    metadata.loc[:, 'imgstore'] = metadata['imgstore_name'].apply(
        lambda x: x.split('/')[1])

    # combine with annotations
    metadata_annotated = metadata.merge(
        matched_long, on=['imgstore', 'well_name'], how='outer')
    metadata_annotated.drop(columns='imgstore', inplace=True)

    # drop unannotated wells
    # metadata_annotated = metadata_annotated[
    #     metadata_annotated.well_label.notna()].reset_index(drop=True)
    metadata_annotated = metadata_annotated.dropna(
        subset=['well_label']).reset_index(drop=True)

    metadata_annotated.to_csv(saveto, index=False)

    return metadata_annotated


def print_elements_in_a_not_in_b(
        a, b, a_name='the first list', b_name='the second list',
        is_warning=True):
    """
    print_elements_in_a_not_in_b: print elements of the first iterable that
        are not in the second
    """
    in_a_not_in_b = list(set(a) - set(b))
    if len(in_a_not_in_b) > 0:
        msg = (
            f'These {len(in_a_not_in_b)} items '
            + f'are in {a_name} but not in {b_name}\n'
            + '\n'.join([str(x) for x in in_a_not_in_b])
            )
        if is_warning:
            warnings.warn(msg)
        else:
            print(msg)
    return


def update_metadata_with_wells_annotations(
        aux_dir, saveto=None, del_if_exists=False):
    """
    update_metadata_with_wells_annotations Add wells annotations to existing
        metadata file.

        1. load the metadata, and perform some basic checks
        2. load all the annotations found in aux_dir, perform basic checks
        3. compare the metadata and annotations. Evaluate whether to propagate
           the annotations to other bluelight conditions
        4. merge metadata and annotations
        5. write to `saveto` file

    Parameters
    ----------
    aux_dir : pathlib path or string
        path to the AuxiliaryFiles directory in the project
    saveto : pathlib path or string, optional
        where to save the metadata with wells annotations.
        If left to be the default None, the output will be saved to
        aux_dir / wells_updated_metadata.csv
    del_if_exists : bool, optional
        in case the output file already exists, setting this flag to True
        will overwrite the output file. Leaving it to the default False,
        the file will not be overwritten and this script will abort.

    Returns
    -------
    metadata_annotated_df : pandas DataFrame, or None
        Dataframe containing all the info in the metadata file, plus the
        manual annotations. If the function aborts early, it will return None.

    Raises
    ------
    Exception
        This function will error out if the metadata contains empty entries
        for the imgstores column. This really should not happen unless the
        metadata file has pathological issues.
    """

    # input check
    if isinstance(aux_dir, str):
        aux_dir = Path(aux_dir)
    if saveto is None:
        saveto = aux_dir / 'wells_updated_metadata.csv'
    elif isinstance(saveto, str):
        saveto = Path(saveto)

    # check if destination file exists
    if saveto.exists():
        warnings.warn(
            f'Metadata with wells annotations, {saveto}, already exists.')
        if del_if_exists:
            warnings.warn('File will be overwritten.')
            saveto.unlink()
        else:
            warnings.warn(
                'Nothing to do here. If you want to recompile the day metadata'
                ', rename or delete the existing file.')
            return

    # find metadata, checks
    metadata_fname = list(aux_dir.rglob('metadata.csv'))
    if len(metadata_fname) > 1:
        warnings.warn(
            f'More than one metadata file in {aux_dir}: \n' +
            f'{metadata_fname} \naborting.')
        return
    elif len(metadata_fname) == 0:
        warnings.warn(f'no metadata file in {aux_dir}, aborting.')
        return
    else:
        metadata_fname = metadata_fname[0]

    # load all annotations
    wells_annotations_df = import_wells_annotations_in_folder(
        aux_dir, imgstore_name='imgstore')

    # and load the metadata too
    metadata_df = pd.read_csv(metadata_fname)
    if metadata_df['imgstore_name'].isna().any():
        warning_msg = (
            f"There are {metadata_df['imgstore_name'].isna().sum()}"
            + ' NaN values in the `imgstore_name column` in the metadata.\n'
            + 'If this is unexpected, you should check your metadata.'
            )
        warnings.warn(warning_msg)
        metadata_df = metadata_df.dropna(subset=['imgstore_name'])
    # strip the imgstore of the date_yyyymmdd/ part, to compare it with anns df
    metadata_df.loc[:, 'imgstore'] = metadata_df['imgstore_name'].apply(
        lambda x: x.split('/')[1])

    # decision time:
    # 1. were only prestim annotated?
    # 2. does it matter? e.g. are there only prestim in the metadata, weird tho
    # 3. do anns and meta match?

    if wells_annotations_df['imgstore'].str.contains('prestim').all():
        if not metadata_df['imgstore'].str.contains('prestim').all():
            # only prestim were annotated, but also non prestim videos taken
            # then need to align bluelight conditions
            raw_dir = Path(str(aux_dir).replace('AuxiliaryFiles', 'RawVideos'))
            wells_annotations_df = wells_annotations_df.rename(
                columns={'imgstore': 'imgstore_prestim'})
            wells_annotations_df = match_rawvids_annotations(
                raw_dir, wells_annotations_df)

    # if the previous if was not entered, annotations have two extra cols
    wells_annotations_df = wells_annotations_df.drop(
        columns=[
            c for c in ['file_id', 'filename'] if c in wells_annotations_df
            ]
        )


    # now in theory, all videos in metadata should feature in the annotations
    imgstores_in_meta = set(metadata_df['imgstore'])
    imgstores_in_anns = set(wells_annotations_df['imgstore'])
    # import pdb
    # pdb.set_trace()
    print_elements_in_a_not_in_b(
        a=imgstores_in_meta,
        b=imgstores_in_anns,
        a_name='metadata',
        b_name='annotations',
        is_warning=True)
    print_elements_in_a_not_in_b(
        a=imgstores_in_anns,
        b=imgstores_in_meta,
        a_name='annotations',
        b_name='metadata',
        is_warning=True)

    print(f'shape of metadata before merging: {metadata_df.shape}')
    print(f'shape of annotations before merging: {wells_annotations_df.shape}')
    # combine with annotations
    metadata_annotated_df = metadata_df.merge(
        wells_annotations_df,
        on=['imgstore', 'well_name'],
        how='outer'
        )
    print(f'shape of annotated metadata: {metadata_annotated_df.shape}')


    # metadata_annotated_df = metadata_annotated_df.drop(columns='imgstore')

    if metadata_annotated_df['is_bad_well'].isna().any():
        warnings.warn(
            'Some wells do not have an annotation. '
            'Check the metadata and any warning you received above'
            )

    metadata_annotated_df.to_csv(saveto, index=False)

    return metadata_annotated_df


# %% examples of how to use
if __name__ == '__main__':

    AUX_DIR = Path(
        '/Volumes/behavgenom$/Ida/Data/Hydra/DiseaseScreen/AuxiliaryFiles')

    wells_annotated_metadata = update_metadata_with_wells_annotations(
        AUX_DIR, saveto=None, del_if_exists=True)

    # old, DEPRECATED example
    # PROJECT_DIR = Path('/Volumes/behavgenom$/Ida/Data/Hydra/DiseaseScreen')

    # wells_annotations_df = import_wells_annotations_in_folder(
    #     PROJECT_DIR / 'AuxiliaryFiles')

    # matched_videos_annotations = match_rawvids_annotations(
    #     PROJECT_DIR / 'RawVideos', wells_annotations_df)

    # wells_annotated_metadata = update_metadata(
    #     PROJECT_DIR / 'AuxiliaryFiles',
    #     matched_videos_annotations,
    #     del_if_exists=True
    #     )
