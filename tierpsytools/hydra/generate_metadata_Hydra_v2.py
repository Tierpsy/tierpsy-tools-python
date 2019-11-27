#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:56:38 2019

@author: ibarlow
"""

""" Function to collate well numbers from features summaries, filenames and 
fuse with the robot.csv. Make sure only one feat summary and filename per folder"""

from pathlib import Path
import pandas as pd
import re
import json
import sys

sys.path.insert(0, '/Users/ibarlow/tierpsy_luigi/tierpsy-tracker')

from tierpsy.analysis.split_fov.helper import CAM2CH_df, UPRIGHT_96WP#,parse_camera_serial, serial2rigchannel

#global variables and regular expressions
set_no = r"(?<=run|set)\d{1,}"
date = r"(?=\d{8}_)\d{8}"
camera = r"(?<=20\d{6}\_\d{6}\.)\d{8}"
meta_index = ['date_yyyymmdd',
              'run_number',
              'well_name',
              'instrument_name']

#%%
def read_json_data(project_directory,dates_to_analyse):
    # add in extracting out the temperature and humidity data
    extra_jsons = list(Path(project_directory).rglob('*extra_data.json'))
    
    if len(extra_jsons)>0:
        json_metadata = pd.DataFrame()
        for e in extra_jsons:
            e= str(e)
            _date = int(re.findall(date, e)[0])
            if _date in dates_to_analyse:
                _set = re.findall(set_no, e, re.IGNORECASE)[0]
                _camera = re.findall(camera,e)[0]
                _rig = HYDRA2CAM_DF.columns[(HYDRA2CAM_DF == _camera).any(axis=0)][0] 
                with open(e) as fid:
                    extras = json.load(fid)
                    for t in extras:
                        json_metadata = json_metadata.append(pd.concat([pd.DataFrame.from_records([
                                {'run_number' : int(_set),
                                 'date_yyyymmdd':_date,
                                 'camera_no': _camera,
                                 'filename': e,
                                 'filedir': Path(e).parent,
                                 'instrument_name':_rig}]), pd.DataFrame(pd.Series(t)).transpose()], axis=1), ignore_index=True, sort=True) 
    
    return json_metadata

def generate_metadata(project_directory, json_metadata):
    """This function goes through the project directory to compile metadata suitable for
    use with the features and filenames summaries
    
    Input:
        project_directory = Pathlib Path to directory for all the experiments. Should contain
        the features_summaries, filenames_summaries and the day_metadata.csv
    
        json_metadata = Bool to determine whether to extract out he .json temp and 
        humidity data
    """
    
    metadata_output_fname = project_directory / 'compiled_metadata.csv'
    
    if metadata_output_fname.exists():
        print ('Metadata file already exists for this project')
        return
    
    else:
        feature_summary_file = list(Path(project_directory).rglob('features_summary*'))#[0]
        filename_summary_file = list(Path(project_directory).rglob('filenames_summary*'))#[0]
        
        #import the metadata files
        input_metadata_files = list(Path(project_directory).rglob('*hydra_robot_metadata.csv'))#[0]
        
        #import metadata
        metadata_in = []
        for f in input_metadata_files:    
            metadata_in.append(pd.read_csv(f,index_col=False))
        
        #concatenate together and drop filename
        metadata_in = pd.concat(metadata_in, sort=True)
        metadata_in.drop(columns = 'filename',inplace=True)
        dates_to_analyse = list(metadata_in.date_yyyymmdd.unique())
        
        print ('Compiling metadata for {}'. format (dates_to_analyse))
    
        #make multi-level index
        metadata_in.index = pd.MultiIndex.from_arrays([metadata_in.date_yyyymmdd,
                                                      metadata_in.run_number,
                                                      metadata_in.well_name,
                                                      metadata_in.instrument_name],
                                                      names = meta_index)
        metadata_in.drop(columns = meta_index,inplace=True)
                 
        #extract from the filename the setnumber, date, camera number and then from file extract well names
        metadataMat = pd.concat([pd.read_csv(feature_summary_file[0], index_col='file_id')[['well_name']],
                             pd.read_csv(filename_summary_file[0],index_col='file_id')],join='outer',axis=1)
        metadataMat.reset_index(drop=False,inplace=True)
        
        #now loop through the rows to extract out the associated wells
        metadata_extract = pd.DataFrame()
        for i,r in metadataMat.iterrows():
            _date = int(re.findall(date, r.file_name)[0])
            if _date in dates_to_analyse:
                _set = re.findall(set_no, r.file_name, re.IGNORECASE)[0]
                _camera = re.findall(camera,r.file_name)[0]
                _rig = HYDRA2CAM_DF.columns[(HYDRA2CAM_DF == _camera).any(axis=0)][0]       
               
                metadata_extract = metadata_extract.append(r.append(pd.Series({'run_number': int(_set),
                                                                              'date_yyyymmdd' : _date,
                                                                              'camera_number' : _camera,
                                                                              'instrument_name' : _rig})),
                                                                    ignore_index=True, sort=True)
              
        metadata_extract.index = pd.MultiIndex.from_arrays([metadata_extract.date_yyyymmdd,
                                                            metadata_extract.run_number,
                                                            metadata_extract.well_name,
                                                            metadata_extract.instrument_name],
                                                            names = meta_index)
        metadata_extract.drop(columns = meta_index,inplace=True)
        
        #concatenate together so that can merge but keep empty wells using join=outer
        metadata_concat = pd.concat([metadata_in, metadata_extract], axis=1, join='outer', sort=True)
        metadata_concat.reset_index(drop=False, inplace=True)
           
        #fill in the file_names for missing wells
        #missing wells
        missing_well_index = metadata_concat[metadata_concat['file_name'].isna()].index.values
        for well in missing_well_index:
            poss_cameras = CAM2CH_df[CAM2CH_df.rig==metadata_concat.loc[well]['instrument_name']] #according to multilevel index
            channel = [col[0] for col in UPRIGHT_96WP if (UPRIGHT_96WP[col]==metadata_concat.loc[well]['well_name']).any()][0]
            camera_serial = (poss_cameras.camera_serial[poss_cameras['channel']==channel]).values[0]
            _date = metadata_concat.loc[well]['date_yyyymmdd']
            poss_file = [f for f in metadata_concat.file_name if str(camera_serial) in str(f) and 'run{}'.format(metadata_concat.loc[well]['run_number']) in str(f) and str(_date) in str(f)]
            
            metadata_concat.loc[well,'file_name'] = list(set(poss_file))[0]
            metadata_concat.loc[well,'camera_number'] = camera_serial
            
        #save to csv
        metadata_concat.to_csv(metadata_output_fname, index=False)
    
        if json_metadata:
            print ('extracting out the .json metadata')
            json_metadata = read_json_data(project_directory, dates_to_analyse)
            # summarise json metadata by export metadata 
            json_metadata.to_csv(project_directory / 'extra_data.csv')
        
            return metadata_concat, json_metadata
        
        else:
            return metadata_concat

if __name__=='__main__':
#    project_directory = sys.argv[0]
#    metadata_file = sys.argv[1]
    
    project_directory = Path('/Volumes/behavgenom$/Ida/Data/Hydra/PilotDrugExps')
#    metadata_file = '/Volumes/behavgenom$/Luigi/Data/LoopBio_tests/metadata.csv'
    
    generate_metadata(project_directory, json_metadata=False)    