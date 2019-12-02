#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:25:41 2019

@author: em812
"""
import pandas as pd
from pathlib import Path
from tierpsytools.hydra.hydra_helper import exract_randomized_by,rename_out_meta_cols
import warnings
import numpy as np

#%%
def merge_robot_metadata (sourceplates_file,randomized_by='column',saveto=None,del_if_exists=False):
    """
    Function that imports the robot runlog and the associated source plates 
    for a given day of experiments and uses them to compile information of 
    drugs in the destination plates
    param:
        sourceplates_file: path to sourceplates_file `YYYYMMDD_sourceplates.csv`
        randomized_by: How did the robot randomize the wells from the source plate to the destination plates?
            options: 'column'/'source_column' = shuffled columns,
            'row'/'source_row' = shuffled rows, 'well'/'source_well' = shuffled well-by-well
            The parameter randomized_by is expected to be a field in the sourceplates file.
    return:
        robot related metadata for the given day of experiments as dataframe
    """
    if saveto is None:
        date = sourceplates_file.stem.split('_')[0]
        saveto = Path(sourceplates_file).parent / (date+'_robot_metadata.csv')
        
    # check if file exists
    if (saveto is not False) and (saveto.exists()):
        if del_if_exists:
            warnings.warn('Robot metadata file {} already exists. File will be overwritten.'.format(saveto))
            saveto.unlink()
        else:
            warnings.warn('Robot metadata file {} already exists. Nothing to do here. If you want to recompile the robot metadata, rename or delete the exisiting file.'.format(saveto))
            return
    
    # required fields in sourceplates file
    sourceplate_cols = ['source_plate_id','robot_runlog_filename','source_robotslot'].extend([randomized_by])
    
    # import the sourceplates
    sourceplates = pd.read_csv(sourceplates_file,index_col=False)
    
    # check if sourceplate dataframe fullfills requirements
    missing_cols = [col for col in sourceplate_cols if col not in sourceplates.columns]
    if len(missing_cols)>0:
        raise KeyError('Field(s) {} do not exist in sourceplates file. This/These field(s) is/are required.'.format(missing_cols))
    for plate in sourceplates['source_plate_id'].unique():
        runlogs = sourceplates.loc[sourceplates['source_plate_id']==plate,'robot_runlog_filename'].unique()
        if runlogs.shape[0]>1:
            raise ValueError('Multiple robot runlogs were defined for the same source plate.')
    
    # read each robot log and compile metadata
    robot_metadata = []
    for n_log,log_file in enumerate(sourceplates['robot_runlog_filename'].unique()):
        
        # get sourceplates lines linked to this robot log
        source = sourceplates.loc[sourceplates['robot_runlog_filename']==log_file,:]
        # get only unique rows of source_robotslot - source_plate_id
        source_map = source[['source_robotslot','source_plate_id']].drop_duplicates()
        # assert that a signle source_robotslot is defined for each single source_plate_id
        assert source_map['source_plate_id'].unique().shape[0] == source_map.shape[0]
        assert source_map['source_robotslot'].unique().shape[0] == source_map.shape[0]
        
        # read robotlog data
        robotlog = pd.read_csv(log_file)
        # keep only data for source_slots with drugs (drop water source_slots)
        robotlog = robotlog[robotlog['source_slot'].isin(source['source_robotslot'])]
        # assign robot runlog id
        robotlog['robot_runlog_id'] = n_log+1
        # extract the column number or the row number from the well number for mapping
        # if the robot randomized based on columns or rows
        robotlog = exract_randomized_by(robotlog,randomized_by)
        # add source_plate_id based on unique source_plate_id - source_slot mapping obtained from sourceplates file
        robotlog['source_plate_id'] = robotlog['source_slot'].map(dict(source_map.values))
        
        # merge all sourceplate data with robot runlog data based on source_plate_id and randomized_by
        out_meta = pd.merge(source,robotlog,how='outer',left_on=['source_plate_id',randomized_by],right_on=['source_plate_id',randomized_by])
        # get unique imaging_plate_id
        out_meta['imaging_plate_id'] = out_meta[['source_plate_id','destination_slot']].apply(lambda x: 'rr{0}_sp{1}_ds{2}'.format(n_log+1,*x),axis=1)
        
        # clean up and rename columns in out_meta
        # - sort rows for readability
        out_meta = out_meta.sort_values(by=['source_plate_id','destination_slot', randomized_by]).reset_index(drop=True)
        # - drop duplicate source_slot info
        assert np.all(out_meta['source_slot']==out_meta['source_robotslot'])
        out_meta = out_meta.drop(labels='source_slot',axis=1)
        # - rename column field for interpretability
        out_meta = rename_out_meta_cols(out_meta)
        # - rearrange columns for readability
        leading_cols = ['imaging_plate_id','source_plate_id','destination_slot']
        end_cols = ['robot_runlog_id','robot_runlog_filename']
        out_meta = out_meta[leading_cols+list(out_meta.columns.difference(leading_cols+end_cols))+end_cols]
        
        # append to list
        robot_metadata.append(out_meta)
    
    # concatenate data from all the robot runlogs
    robot_metadata = pd.concat(robot_metadata,axis=0)
    
    if saveto is not False:
        robot_metadata.to_csv(saveto,index=None)
        
    return robot_metadata


#%%
# STEP 2
def get_camera_serial(metadata,n_wells=96):
    """
    Get the camera serial number from the well_name and instrument_name
    param:
        metadata = dataframe with day metadata
    return:
        out_metadata = day metadata dataframe including camera serial
    """
    from tierpsytools.hydra import CAM2CH_df,UPRIGHT_96WP
    
    if n_wells != 96:
        raise ValueError('Only 96-well plates supported at the moment.')
    
    channels = ['Ch{}'.format(i) for i in range(1,7,1)]

    WELL2CH = []
    for ch in channels:
        chdf = pd.DataFrame(UPRIGHT_96WP[ch].values.reshape(-1,1),columns=['well_name'])
        chdf['channel'] = ch
        WELL2CH.append(chdf)
    WELL2CH = pd.concat(WELL2CH,axis=0)
    
    WELL2CAM = pd.merge(CAM2CH_df,WELL2CH,how='outer',on='channel').sort_values(by=['rig','channel','well_name'])
    WELL2CAM = WELL2CAM.rename(columns={'rig':'instrument_name'})

    out_metadata = pd.merge(metadata,WELL2CAM[['instrument_name','well_name','camera_serial']],how='outer',left_on=['instrument_name','well_name'],right_on=['instrument_name','well_name'])
    assert out_metadata.shape[0] == metadata.shape[0]
    
    return out_metadata

    
def add_imgstore_name(metadata,aux_day_dir,raw_root=None,n_wells=96):
    """
    Add the imgstore name of the hydra videos to the day metadata dataframe
    param:
        metadata = dataframe with metadata for a given day of experiments. See README.md for details on fields.
        n_wells = number of wells in imaging plate (only 96 supported at the moment)
    return:
        out_metadata = metadata dataframe with imgstore_name added
    """    
    
    if raw_root is None:
        raw_day_dir = Path(str(aux_day_dir).replace('AuxiliaryFiles','RawVideos'))
    else:
        raw_day_dir = Path(raw_root) / aux_day_dir.stem
    
    # get add camera serial number
    metadata = get_camera_serial(metadata,n_wells=n_wells)
    
    # get imgstore full path in raw videos
    MAP2PATH = metadata[['imaging_run_number','camera_serial']].drop_duplicates()
    MAP2PATH['run_name'] = metadata['imaging_run_number'].apply(lambda x: 'run{}'.format(x))
    #print('There are {} videos expected from metadata in {}.\n'.format(MAP2PATH.shape[0],aux_day_dir))
    
    file_list = [file for file in raw_day_dir.rglob("metadata.yaml")]
    #print('There are {} raw videos found in {}.\n'.format(len(file_list),raw_day_dir))
    
    if len(file_list)!=MAP2PATH.shape[0]:
        warnings.warn("\n\nThe number of videos found does not match the number of videos expected. The compilation of metadata for day {} will continue.\n\n".format(aux_day_dir.stem))
    
    MAP2PATH['imgstore_name'] = MAP2PATH[['run_name','camera_serial']].apply(lambda x: [file for file in file_list if np.all([str(ix) in str(file) for ix in x])],axis=1)
    
    # checks
    not_unique = MAP2PATH['imgstore_name'].apply(lambda x: True if len(x)>1 else False).values
    not_found = MAP2PATH['imgstore_name'].apply(lambda x: True if len(x)==0 else False).values
    if np.sum(not_unique)>0:
        not_unique = MAP2PATH.loc[not_unique,:]
        for i,row in not_unique.iterrows():
            warnings.warn('\n\nMore than one video found for day {}, run{}, camera {}.\n\n'.format(aux_day_dir.stem,*row.values))
    if np.sum(not_found)>0:
        not_found = MAP2PATH.loc[not_found,:]
        for i,row in not_found.iterrows():
            warnings.warn('\n\nNo video found for day {}, run {}, camera {}.\n\n'.format(aux_day_dir.stem,*row.values))
    
    # keep only imgstore_name (experiment_day_dir/imgstore_name_dir)
    MAP2PATH['imgstore_name'] = MAP2PATH['imgstore_name'].apply(lambda x: '/'.join(x[0].parts[-3:-1]) if len(x)==1 else np.nan)
    
    # merge dataframes to store imgstore_name for each metadata row
    out_metadata = pd.merge(metadata,MAP2PATH[['imaging_run_number','camera_serial','imgstore_name']],how='outer',on=['imaging_run_number','camera_serial'])
    assert out_metadata.shape[0] == metadata.shape[0]
    
    return out_metadata
    

def get_day_metadata(imaging_plate_metadata, manual_metadata_file, merge_on=['imaging_plate_id'], n_wells=96, saveto=None,del_if_exists=False):
    """ 
    Incorporates the robot metadata and the manual metadata of the hydra rigs
    to get all the metadata for a given day of experiments.
    Also, it adds the imgstore_name of the hydra vidoes.
    Input:
        imaging_plate_metadata: dataframe containing the metadata for all the wells of each imaging plate
                (If the robot was used, then this is the robot_metadata obtained by the robot_to_metadata.merge_robot_metadata function)
        manual_metadata: .csv file with the details of rigs and runs (see README
                          for minimum required fields)
        merge_on: column(s) in common in imaging_plate_metadata and manual_metadata, that can be used to merge them
        n_wells: number of wells in imaging plate
        saveto: csv file path to save the day metadata in (if None, the metadata are saved in the same folder as the manual metadata)
    return:
        metadata: dataframe with day metadata
    """
    #find the date of the hydra experiments
    date_of_runs = manual_metadata_file.stem.split('_')[0]
    
    aux_day_dir = manual_metadata_file.parent
    if saveto is None:
        saveto = aux_day_dir / '{}_day_metadata.csv'.format(date_of_runs)
    
    if metadata_file.exists():
        if  del_if_exists:
            warnings.warn('\n\nMetadata file {} already exists. File will be overwritten.\n\n'.format(saveto))
            metadata_file.unlink()
        else:
            warnings.warn('\n\nMetadata file {} already exists. Nothing to do here.\n\n'.format(saveto))
            return
    
    manual_metadata = pd.read_csv(manual_metadata_file, index_col=False).assign(date_yyyymmdd=date_of_runs)
    
    # merge two dataframes
    metadata = robot_metadata.merge(manual_metadata,how='inner', on=merge_on,indicator=False)
    
    # clean up and rearrange
    metadata.rename(columns={'destination_well':'well_name'}, inplace=True)
    
    # add imgstore name
    metadata = add_imgstore_name(metadata,aux_day_dir,n_wells=n_wells)
    
    print('Saving metadata file: {} '.format(metadata_file))        
    metadata.to_csv(metadata_file, index=False)
    
    return metadata


#%%
# STEP 3:
def concatenate_days_metadata(aux_root_dir,list_days=None,saveto=None):
    """
    Reads all the yyyymmdd_day_metadata.csv files from the different days of experiments
    and creates a full metadata file all_metadata.csv
    param:
        aux_root_dir: root AuxiliaryFiles directory containing all the folders for the individual days of experiments
        list_days: list of folder names (dates) to read from. If None (default), it reads all the subfolders in the root_dir.
        saveto: filename to save all compiled metadata to
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
        auxfiles = aux_root_dir.joinpath(day).glob('*_day_metadata.csv')
        
        # checks
        if len(auxfiles) > 1:
            is_ok = np.zeros(len(auxfiles)).dtype(bool)
            for ifl,file in enumerate(auxfiles):
                if len(file.stem.replace('_day_metadata',''))==8:
                    is_ok[ifl]=True
            if np.sum(is_ok)==1:
                auxfiles = [file for ifl,file in enumerate(auxfiles) if is_ok[ifl]]
            else:
                raise ValueError('More than one *_day_metatada.csv files found in {}. Compilation of metadata is aborted.'.format())
        
        meta_files.append(auxfiles[0])
        
    all_meta = []
    for file in meta_files:
        all_meta.append(pd.read_csv(file))
    
    all_meta = all_meta.concat(all_meta,axis=0)
    all_meta.to_csv(saveto,index=False)
    
    return

#%%
if __name__ == '__main__':
    # Example 1:
    # Input
    day_root_dir = Path('/Users/em812/Data/Hydra_pilot/AuxiliaryFiles/20191108_tierpsytools_dev')
    sourceplate_file = day_root_dir / '20191107_sourceplates.csv'
    manual_meta_file = day_root_dir / '20191108_manual_metadata.csv'
    
    # Save to
    #robot_metadata_file = day_root_dir.joinpath('20191107_robot_metadata.csv')
    metadata_file = day_root_dir.joinpath('20191108_day_metadata.csv')
    
    # Run
    robot_metadata = merge_robot_metadata(sourceplate_file, saveto=False)
    day_metadata = get_day_metadata(robot_metadata, manual_meta_file, saveto=metadata_file)
        
    # Example 2:
    # Input
    aux_root_dir = Path('/Volumes/behavgenom$/Ida/Data/Hydra/PilotDrugExps/AuxiliaryFiles')
    day_root_dirs = [d for d in aux_root_dir.glob("*") if d.is_dir()]
    
    sourceplate_files = [[file for file in d.glob('*_sourceplates.csv')] for d in day_root_dirs]
    manual_meta_files = [[file for file in d.glob('*_manual_metadata.csv')] for d in day_root_dirs]
    
    # Saveto
    metadata_files = [d.joinpath('{}_day_metadata.csv'.format(d.stem)) for d in day_root_dirs]
    
    # Run
    for day,source,manual_meta,saveto in zip(day_root_dirs,sourceplate_files,manual_meta_files,metadata_files):
        if len(source)!=1:
            print('There is not a unique sourceplates file in day {}. Metadata cannot be compiled'.format(day))
            continue
        if len(manual_meta)!=1:
            print('There is not a unique manual_metadata file in day {}. Metadata cannot be compiled'.format(day))
            continue
        robot_metadata = merge_robot_metadata(source[0], saveto=False)
        day_metadata = get_day_metadata(robot_metadata, manual_meta[0], saveto=saveto[0])
    