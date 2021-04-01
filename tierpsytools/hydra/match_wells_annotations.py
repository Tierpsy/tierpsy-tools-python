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
from tierpsytools.hydra.match_bluelight_videos import match_bluelight_videos_in_folder

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
        DESCRIPTION. The default is ['/filenames_df',                                          '/wells_annotations_df'].

    imgstore_name: default is imgstore_prestim

    Returns
    -------
    annotations : TYPE
        DESCRIPTION.

    """
    with pd.HDFStore(annotations_file) as fid:
        _fnames = fid[table_names[0]]
        _markings = fid[table_names[1]]
    annotations_df = _fnames.merge(_markings[['file_id',
                                            'well_label',
                                            'well_name']],
                                    on='file_id',
                                    # right_index=True,
                                    validate='one_to_many')
    annotations_df['imgstore_prestim'] = annotations_df.filename.apply(
                                        lambda x: x.split('/')[0])
    return annotations_df

def import_wells_annoations_in_folder(aux_dir,
                                      search_string='*wells_annotations.hdf5'):
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

    annotations_files = list(Path(aux_dir).rglob(search_string))
    annotations_df=[]
    for f in annotations_files:
        annotations_df.append(import_wells_annotations(f))
    annotations_df = pd.concat(annotations_df)
    annotations_df.reset_index(drop=True,
                               inplace=True)
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
        DESCRIPTION. The default is ['imgstore_prestim',                                               'imgstore_bluelight',                                               'imgstore_poststim'].

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
                                                         'well_label'],
                                                value_vars=bluelight_names,
                                                value_name='imgstore')
    matched_long.drop(columns=['variable'],
                      inplace=True)
    matched_long['is_bad_well'] = matched_long['well_label'] != 1

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
    metadata_fname = list(Path(aux_dir).rglob('metadata.csv'))

    if len(metadata_fname)>1:
        warnings.warn('More than one metadata file in this directory: \n' +
                      '{} \n'.format(metadata_fname) +\
                          'aborting')
        return None

    # saving and overwriting checks
    if saveto is None:
        saveto = aux_dir / 'wells_updated_metadata.csv'

    # check if file exists
    if (saveto is not False) and (saveto.exists()):
        if del_if_exists:
            warnings.warn('Wells annotations file, {} already'.format(saveto)
                          + 'exists. File will be overwritten.')
            saveto.unlink()
        else:
            warnings.warn('Wells annotations file, {} already '.format(saveto)
                          + 'exists. Nothing to do here. If you want to '
                          + 'recompile the day metadata, rename or delete the '
                          + 'exisiting file.')
            return None

    #read metadata
    metadata = pd.read_csv(metadata_fname[0])
    if metadata['imgstore_name'].isna().sum() > 0:
        print('Nan values in imgstore names')
        metadata = metadata[metadata['imgstore_name'].notna()]
    metadata.loc[:,'imgstore'] = metadata['imgstore_name'].apply(
                                lambda x: x.split('/')[1])
    #combine with annotations
    metadata_annotated = metadata.merge(matched_long,
                                 on=['imgstore',
                                     'well_name'],
                                 how='outer')
    metadata_annotated.drop(columns='imgstore',
                        inplace=True)
    # drop unannotated wells
    metadata_annotated = metadata_annotated[metadata_annotated.well_label.notna()
                                      ].reset_index(drop=True)

    metadata_annotated.to_csv(saveto,
                              index=False)

    return metadata_annotated

#%% examples of how to use
if __name__=='__main__':
    PROJECT_DIR = Path('/Volumes/behavgenom$/Ida/Data/Hydra/DiseaseScreen')

    wells_annotations_df = import_wells_annoations_in_folder(PROJECT_DIR / 'AuxiliaryFiles')

    matched_videos_annoations = match_rawvids_annotations(PROJECT_DIR / 'RawVideos',
                                                          wells_annotations_df)

    wells_annotated_metadata = update_metadata(PROJECT_DIR / 'AuxiliaryFiles',
                                               matched_videos_annoations,
                                               del_if_exists=True)
