#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:31:14 2020

@author: ibarlow

Helper functions for reading the disease data and making strain specific
plots
"""
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
# from scipy import stats

from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions
from tierpsytools.preprocessing.filter_data import drop_ventrally_signed, filter_nan_inf, cap_feat_values, feat_filter_std

# CONTROL_STRAIN = 'N2'
DATES_TO_DROP = '20200626'
BAD_FEAT_THRESH = 3 # 3 standard deviations away from the mean
# BAD_FEAT_FILTER = 0.1 # 10% threshold for removing bad features
# BAD_WELL_FILTER = 0.3 # 30% threshold for bad well

BAD_FEAT_FILTER = 0.05 # 5% threshold for removing bad features
BAD_WELL_FILTER = 0.5 # 50% threshold for bad well


STIMULI_ORDER = {'prestim':1,
                 'bluelight':2,
                 'poststim':3}

BLUELIGHT_WINDOW_DICT = {0:[55,'prelight',1],
                        1: [70, 'bluelight',1],
                        2: [80, 'postlight',1],
                        3: [155, 'prelight',2],
                        4: [170, 'bluelight',2],
                        5: [180, 'postlight',2],
                        6: [255, 'prelight',3],
                        7: [270, 'bluelight',3],
                        8: [280, 'postlight',3]}

def drop_nan_worms(feat, meta, saveto, export_nan_worms=False):

    # remove (and check) nan worms
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]
    if export_nan_worms:
        nan_worms.to_csv(saveto / 'nan_worms.csv',
                          index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)

    return feat, meta

def read_disease_data(feat_file, fname_file, metadata_file, drop_nans=True, export_nan_worms=False):
    """

    Parameters
    ----------
    feat_file : TYPE
        DESCRIPTION.
    fname_file : TYPE
        DESCRIPTION.
    metadata_file : TYPE
        DESCRIPTION.
    export_nan_worms : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    feat : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.

    """
    # feat = pd.read_csv(feat_file,
    #                    comment='#')
    # fname = pd.read_csv(fname_file,
    #                     comment='#')
    # meta = pd.read_csv(metadata_file, index_col=None)
    # meta['imaging_date_yyymmdd'] = pd.to_datetime(meta.imaging_date_yyyymmdd,
    #                                               format='%Y%m%d').dt.date
    
    # assert meta.worm_strain.unique().shape[0] == meta.worm_gene.unique().shape[0]

    feat, meta = read_hydra_metadata(feat_file,
                                     fname_file,
                                     metadata_file)
    meta['date_yyyymmdd'] = pd.to_datetime(meta.date_yyyymmdd,
                                                  format='%Y%m%d').dt.date
    
    #assert meta.worm_strain.unique().shape[0] == meta.worm_gene.unique().shape[0]
    
    feat, meta = align_bluelight_conditions(feat,
                                            meta,
                                            how='inner') #removes wells that don't have all 3 conditions
    if drop_nans:
        feat, meta = drop_nan_worms(feat, meta, saveto=feat_file.parent)

    return feat, meta


def select_strains(candidate_gene, control_strain, meta_df, feat_df=pd.DataFrame()):
    """

    Parameters
    ----------
    strains : TYPE
        DESCRIPTION.
    feat_df : TYPE
        DESCRIPTION.
    meta_df : TYPE
        DESCRIPTION.
    control_strain : TYPE
        DESCRIPTION.

    Returns
    -------
    feat_df : TYPE
        DESCRIPTION.
    meta_df : TYPE
        DESCRIPTION.

    """

    gene_list = [g for g in meta_df.worm_gene.unique() if g != control_strain]
    #gene_list = [g for g in gene_list if 'myo' not in g and 'unc-54' not in g]
    gene_list.sort()
        
    # if len(candidate_gene) <=1:
    #     idx = [c for c,g in list(enumerate(gene_list)) if  g==candidate_gene]
    # else:
    if control_strain not in candidate_gene:
        idx = [gene_list.index(item) for item in candidate_gene]
    else:
        idx=[]
        
    locs = list(meta_df.query('@candidate_gene in worm_gene').index)
    # date_to_select = meta_df.loc[locs]['date_yyyymmdd'].unique()
    # N2_locs = list(meta_df.query('@date_to_select in date_yyyymmdd and @control_strain in worm_gene').index)
    N2_locs = list(meta_df.query('@control_strain in worm_gene').index)
    locs.extend(N2_locs)

    #Only do analysis on the disease strains
    meta_df = meta_df.loc[locs,:]
    if feat_df.empty:
        return meta_df, idx, gene_list
    else:
        feat_df = feat_df.loc[locs,:]

        return feat_df, meta_df, idx, gene_list


def filter_features(feat_df, meta_df, dates_to_drop=DATES_TO_DROP):
    """

    Parameters
    ----------
    feat_df : TYPE
        DESCRIPTION.
    meta_df : TYPE
        DESCRIPTION.

    Returns
    -------
    feat_df : TYPE
        DESCRIPTION.
    meta_df : TYPE
        DESCRIPTION.
    featsets : TYPE
        DESCRIPTION.

    """
    imgst_cols = [col for col in meta_df.columns if 'imgstore_name' in col]
    miss = meta_df[imgst_cols].isna().any(axis=1)

    # remove data from dates to exclude
    bad_date = meta_df.date_yyyymmdd == float(dates_to_drop)

    # remove wells annotated as bad

    good_wells_from_gui = meta_df.is_bad_well == False
    feat_df = feat_df.loc[good_wells_from_gui & ~bad_date & ~miss,:]
    meta_df = meta_df.loc[good_wells_from_gui & ~bad_date & ~miss,:]

    # remove features and wells with too many nans and std=0
    feat_df = filter_nan_inf(feat_df,
                             threshold=BAD_FEAT_FILTER,
                             axis=0)
    feat_df = filter_nan_inf(feat_df,
                             threshold=BAD_WELL_FILTER,
                              axis=1)

    feat_df = feat_filter_std(feat_df)
    feat_df = cap_feat_values(feat_df)
    feat_df = drop_ventrally_signed(feat_df)

    meta_df = meta_df.loc[feat_df.index,:]
    # feature sets
     # abs features no longer in tierpsy
    pathcurvature_feats = [x for x in feat_df.columns if 'path_curvature' in x]
    #remove these features
    feat_df = feat_df.drop(columns=pathcurvature_feats)

    featlist = list(feat_df.columns)
    # for f in featsets.keys():
    #     featsets[f] = [x for x in feat.columns if f in x]
    #     featsets[f] = list(set(featsets[f]) - set(pathcurvature_feats))

    featsets={}
    for stim in STIMULI_ORDER.keys():
        featsets[stim] = [f for f in featlist if stim in f]
    featsets['all'] = featlist

    return feat_df, meta_df, featsets


def make_colormaps(gene_list, featlist, CONTROL_STRAIN, idx=[], candidate_gene=None):
    """

    Parameters
    ----------
    gene_list : TYPE
        DESCRIPTION.
    idx : TYPE
        DESCRIPTION.
    candidate_gene : TYPE
        DESCRIPTION.
    CONTROL_STRAIN : TYPE
        DESCRIPTION.
    STIMULI_ORDER : TYPE
        DESCRIPTION.
    featlist : TYPE
        DESCRIPTION.

    Returns
    -------
    strain_cmap : TYPE
        DESCRIPTION.
    strain_lut : TYPE
        DESCRIPTION.
    stim_cmap : TYPE
        DESCRIPTION.
    feat_lut : TYPE
        DESCRIPTION.

    """

    cmap = list(np.flip((sns.color_palette('cubehelix',
                                           len(gene_list)*2+6))[3:-4:2]))
    N2_cmap = (0.6, 0.6, 0.6)

    strain_lut = {}
    # strain_lut[CONTROL_STRAIN] = CONTROL_DICT[CONTROL_STRAIN]

    if candidate_gene is not None:
        for c,g in enumerate(candidate_gene):
            strain_lut[g] = cmap[idx[c]]

        # dict(zip([candidate_gene,
        #           CONTROL_STRAIN],
        #          strain_cmap))

    cmap.append(N2_cmap)
    gene_list.append(CONTROL_STRAIN)
    strain_lut.update(dict(zip(gene_list,
                          cmap)))

    stim_cmap = sns.color_palette('Pastel1',3)
    stim_lut = dict(zip(STIMULI_ORDER.keys(), stim_cmap))

    if len(featlist)==0:
        return strain_lut, stim_lut


    feat_lut = {f:v for f in featlist for k,v in stim_lut.items() if k in f}
    return strain_lut, stim_lut, feat_lut


# def write_ordered_features(clusterfeats, saveto):
#     Path(saveto).touch(exist_ok=False)
#     with open(saveto, 'w') as fid:
#         for l in clusterfeats:
#             fid.writelines(l + ',\n')

#     return


def find_window(fname):
    import re
    window_regex = r"(?<=_window_)\d{0,9}"
    window = int(re.search(window_regex, str(fname))[0])
    return window

def strain_gene_dict(meta):
    """


    Parameters
    ----------
    meta : TYPE
        DESCRIPTION.

    Returns
    -------
    strain_dict : TYPE
        DESCRIPTION.

    """
    strain_dict = {r.worm_strain : r.worm_gene for
             i,r in meta[['worm_strain',
                        'worm_gene']].drop_duplicates().iterrows()
           }
    return strain_dict

def long_featmap(feat, meta, stim=['prestim', 'bluelight', 'poststim']):
    """
    Convert wide-form feature dataframe to longform

    Parameters
    ----------
    feat : TYPE
        DESCRIPTION.

    meta

    Returns
    -------
    None.

    """
    featlist = list(feat.columns)
    meta_cols = list(meta.columns)
    long_featmat = []
    long_meta = []
    for st in stim:
        stim_list = [f for f in featlist if st in f]
        _featmat = pd.DataFrame(data=feat.loc[:,stim_list].values,
                                columns=['_'.join(s.split('_')[:-1])
                                           for s in stim_list],
                                index=feat.index)
        imgst_col = [i for i in meta_cols if all(['imgstore_name' in i, st in i])]

        _meta = pd.concat([_featmat, meta], axis=1)[meta_cols]
        _meta['bluelight'] = st
        _meta['imgstore_name'] = _meta[imgst_col]
        _meta.drop(columns=[i for i in meta_cols if 'imgstore_name_' in i],
                    inplace=True)
        _meta.reset_index(drop=True, inplace=True)
        _featmat.reset_index(drop=True, inplace=True)
        long_featmat.append(_featmat)
        long_meta.append(_meta)

    long_featmat = pd.concat(long_featmat,
                             axis=0)
    long_featmat.reset_index(drop=True,
                             inplace=True)
    long_meta = pd.concat(long_meta,
                          axis=0)
    long_meta.reset_index(drop=True,
                          inplace=True)

    return long_featmat, long_meta

# if __name__ == '__main__':
#     FEAT_FILE = Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/features_summary_tierpsy_plate_20200930_125752.csv')
#     FNAME_FILE = Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/filenames_summary_tierpsy_plate_20200930_125752.csv')
#     METADATA_FILE = Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/AuxiliaryFiles/wells_annotated_metadata.csv')

#     CONTROL_STRAIN = 'N2'
#     CANDIDATE_GENE='cat-2'

#     SAVETO = FEAT_FILE.parent.parent.parent / 'Figures' / 'paper_figures' / CANDIDATE_GENE
#     SAVETO.mkdir(exist_ok=True)
#     feat256_fname = Path('/Users/ibarlow/tierpsy-tools-python/tierpsytools/extras/feat_sets/tierpsy_256.csv')

#     feat, meta = read_disease_data(FEAT_FILE,
#                                    FNAME_FILE,
#                                    METADATA_FILE,
#                                    export_nan_worms=False)

#     feat, meta, idx, gene_list = select_strains(CANDIDATE_GENE,
#                                                 CONTROL_STRAIN,
#                                                 feat,
#                                                 meta)
