#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:25:41 2019

@author: em812
"""
import pandas as pd
from pathlib import Path
<<<<<<< Updated upstream
=======
from tierpsytools.hydra.hydra_helper import exract_randomized_by
from tierpsytools.hydra.hydra_helper import rename_out_meta_cols, explode_df
>>>>>>> Stashed changes
import re
import warnings
import numpy as np
import itertools

#%%
def merge_robot_metadata (sourceplates_file,saveto=None,del_if_exists=False):
    """
    Function that imports the robot runlog and the associated source plates 
    for a given day of experiments and uses them to compile information of 
    drugs in the destination plates
    Input:
        robot_directory - path to robot outputs
        source_plates - path to the source plates file `YYYYMMDD_sourceplates.csv`
    Output:
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
    sourceplate_cols = ['source_plate_id','robot_runlog_filename','source_robotslot']
    
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
        # extract the column number from the well number for mapping (as replicates are by column)
        robotlog['column'] = [int(re.findall(r'\d+', r['source_well'])[0]) for i,r in robotlog.iterrows()]
        # add source_plate_id based on unique source_plate_id - source_slot mapping obtained from sourceplates file
        robotlog['source_plate_id'] = robotlog['source_slot'].map(dict(source_map.values))
        
        # merge all sourceplate data with robot runlog data based on source_plate_id and column
        out_meta = pd.merge(source,robotlog,how='outer',left_on=['source_plate_id','column'],right_on=['source_plate_id','column'])
        # get unique imaging_plate_id
        out_meta['imaging_plate_id'] = out_meta[['source_plate_id','destination_slot']].apply(lambda x: 'rr{0}_sp{1}_ds{2}'.format(n_log+1,*x),axis=1)
        
        # clean up and rename columns in out_meta
        # - sort rows for readability
        out_meta = out_meta.sort_values(by=['source_plate_id','destination_slot', 'column']).reset_index(drop=True)
        # - drop duplicate source_slot info
        assert np.all(out_meta['source_slot']==out_meta['source_robotslot'])
        out_meta = out_meta.drop(labels='source_slot',axis=1)
        # - rename column field for interpretability
        out_meta = out_meta.rename(columns={'column':'source_plate_column'})
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

#%% populate 96 well plates for tracking experiment where plates have been filled with different strains of food or worms

<<<<<<< Updated upstream
def populate_96WPs(source_plates, entire_rows=True, saveto= None, del_if_exists = False):
    """ Input:
        source_plates - path for YYYYMMDD_sourceplates.csv file
        
        Output:
        plate_metadata - one line per well; can be used in get_day_metadata function to compile with manual metadata
            """
    if saveto is None:
        date = source_plates.stem.split('_')[0]
        saveto = Path(source_plates).parent / (date+'_plate_metadata.csv')
=======
def populate_96WPs(
        worm_sorter, entire_rows=True, saveto= None, del_if_exists = False
        ):
    """ 
    Populate 96 well plates for tracking experiment where plates have been 
    filled with different strains of food or worms.
    param:
        source_plates: path for YYYYMMDD_sourceplates.csv file
        
    return:
        plate_metadata: one line per well; can be used in get_day_metadata
            function to compile with manual metadata
    """
>>>>>>> Stashed changes
        
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
<<<<<<< Updated upstream
            warnings.warn('Plate metadata file {} already exists. File will be overwritten.'.format(saveto))
            saveto.unlink()
        else:
            warnings.warn('Plate metadata file {} already exists. Nothing to do here. If you want to recompile the day metadata, rename or delete the exisiting file.'.format(saveto))
            return 
    
    #parameters for the 96WPs    
    n_columns= 12
    column_names = np.arange(1,n_columns+1)
    row_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    well_names = list(itertools.product(row_names, column_names))
    
    #import manual_metadata and source plates
#    manualDF = pd.read_csv(manual_metadata)
    sourceplatesDF = pd.read_csv(source_plates)
    
    #extract out the start and end rows and columns
    sourceplatesDF['start_row'] = [re.findall(r"[A-Z]", r.start_well)[0] for i,r in sourceplatesDF.iterrows()]
    sourceplatesDF['end_row'] = [re.findall(r"[A-Z]", r.end_well)[0] for i,r in sourceplatesDF.iterrows()]
    sourceplatesDF['start_column']=[re.findall(r"(\d{1,2})", r.start_well)[0] for i,r in sourceplatesDF.iterrows()]
    sourceplatesDF['end_column'] = [re.findall(r"(\d{1,2})", r.end_well)[0] for i,r in sourceplatesDF.iterrows()]
   
    #create 96WP template to fill up
=======
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
>>>>>>> Stashed changes
    plate_template = pd.DataFrame()
    plate_template['imaging_plate_row'] = [i[0] for i in well_names]
    plate_template['imaging_plate_column'] = [i[1] for i in well_names]
    plate_template['well_name'] = [i[0] + str(i[1]) for i in well_names]
<<<<<<< Updated upstream
    
    if entire_rows:
        #populate the dataframe
        plate_metadata=[]
        for i,r in sourceplatesDF.iterrows():
            _section = (plate_template[(r.start_row<=plate_template.imaging_plate_row)&
                                       (plate_template.imaging_plate_row<=r.end_row)]).reset_index(drop=True)
            _details = pd.concat(_section.shape[0]*[r.to_frame().transpose()]).reset_index(drop=True)
            plate_metadata.append(pd.concat([_section,
                                             _details],axis=1,sort=True))
        
        plate_metadata = pd.concat(plate_metadata)
        
#        day_metadata = pd.merge(manualDF,
#                               day_metadata,
#                               how='outer',
#                               on= merge_col)
        
        plate_metadata.to_csv(saveto, index=False)
        
        return plate_metadata
    
    else:
        print ('cannot use this function for generating the metadata; fills entire rows only')
        return
=======

    if not entire_rows:
        print ('cannot use this function for generating the metadata;'+
               ' fills entire rows only.')
        return None
    
    # populate the dataframe
    plate_metadata=[]
    for i,r in worm_sorter_df.iterrows():
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
>>>>>>> Stashed changes
    


#%% extra function if using both robot and wormsorter
    
def merge_robot_wormsorter(robot_metadata,
                           plate_metadata,
                           merge_on=['imaging_plate_id']
                           ):
    """ Function for combining the outputs of the above so that concise and
    comprehensive dataframe can be used in get_day_metadata
    Input:
    robot_metadata - output from merge_robot_metadata function
    
    plate_metadata - output form populate_96WP function
    
    Ouput:
    concat_metadata
    """
    concat_metadata = pd.merge(robot_metadata,
                               plate_metadata,
                               on=merge_on,
                               how='outer'
                               )
    return concat_metadata

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
    WELL2CAM= WELL2CAM[WELL2CAM['rig'].isin(metadata['instrument_name'].unique())]
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
    metadata = imaging_plate_metadata.merge(manual_metadata,how='inner', on=merge_on,indicator=False)
    
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
    day_root_dir = Path('/Volumes/behavgenom$/Ida/Data/Hydra/PilotDrugExps/AuxiliaryFiles/20191108_tierpsytools_dev')
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
        
    #%%Example 3:
    day_root_dir = Path('/Volumes/behavgenom$/Ida/Data/Hydra/ICDbacteria/AuxiliaryFiles/20191122')
    sourceplate_file = day_root_dir / '20191122_wormsorter.csv'
    manual_meta_file = day_root_dir / '20191122_manual_metadata.csv'
    metadata_file = day_root_dir.joinpath('20191122_day_metadata.csv')
    
    plate_metadata = populate_96WPs(sourceplate_file,
                                  entire_rows=True,
                                  saveto= None,
                                  del_if_exists = False)
    day_metadata = get_day_metadata(plate_metadata, manual_meta_file, saveto=metadata_file)
<<<<<<< Updated upstream
    
=======
    #%% Example 4
    
    
    concatenated_metadata = merge_robot_wormsorter(robot_metadata, plate_metadata)
>>>>>>> Stashed changes
