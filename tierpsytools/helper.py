#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:16:30 2019

@author: em812
"""

def replace_none(list_object,replace_none_with):
    list_mod = list_object[:]
    for i in range(len(list_object)):
        if list_mod[i] is None:
            list_mod[i]=replace_none_with
    return list_mod
            
def create_dir(new_dir,root_dir=None):
    import os
    
    if root_dir is None:
        full_path = new_dir
    else:
        full_path = os.path.join(root_dir,new_dir)
        
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    return full_path

def get_link_dictionary(df,col1,col2):
    
    if col1 not in df.columns:
        ValueError('column {} not in dataframe'.format(col1))
        return
    if col2 not in df.columns:
        ValueError('column {} not in dataframe'.format(col2))
        return
    
    unique_col1 = df[col1].unique()
    unique_col2 = []
    for i,elm in enumerate(unique_col1):
         tmp = df.loc[df[col1]==elm,col2]
         unique_col2.append(tmp.iloc[0])         
    
    link_dict = dict(zip(unique_col1,unique_col2))
    
    return link_dict


def flatList(mylist):
    flat_list = [item for sublist in mylist for item in sublist]
    return flat_list


def sumLists(list1,list2,delim='_'):
    sum_list = [x1+delim+x2 for (x1,x2) in zip(list1, list2)]
    return sum_list


def copy_files(root,copyroot,pattern,dir_levels_to_keep=None):
    """
    Copy files with specific pattern from a root directory to another location.
    Option to keep subdirectories structure.
    param:
        - root = directory in which to look for pattern
        - copyroot = directory where files will be copied
        - pattern = pattern used in glob to find files to copy
        - dir_levels_to_keep = if it is not None, then subdirectories will be 
            created in copyroot.
    Example:
        We want to copy files:
            './root/dir1/dir2/file1.txt'
            './root/dir1/dir2/file2.txt'
        to './copyroot'.
        -----------
        Input:
            root = './root'
            copyroot = './copyroot'
            pattern = '*/*/*.txt'
        If dir_levels_to_keep=None, then we'll get:
            'copyroot/file1.txt'
            'copyroot/file2.txt'
        If dir_levels_to_keep=1, then we'll get:
            'copyroot/dir2/file1.txt'
            'copyroot/dir2/file2.txt'
        If dir_levels_to_keep=2, then we'll get:
            'copyroot/dir1/dir2/file1.txt'
            'copyroot/dir1/dir2/file2.txt'
        etc..
    """
    from pathlib import Path
    import shutil
    from os.path import join
    from time import time
       
    list_files = [file for file in Path(root).glob(pattern)]
    
    for i,file in enumerate(list_files):
        start_time = time()
        print('copying {} of {}'.format(i,len(list_files)))
        if dir_levels_to_keep is not None:
            to_dir = Path(join(copyroot,*file.parts[-dir_levels_to_keep:-1]))
        to_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(file), join(str(to_dir),file.parts[-1]))
        print('Done in {} sec.'.format(time()-start_time))
        