#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:45:23 2020

@author: ibarlow

timeseries helper functions

"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, 'X:\\Bonnie\\Scripts\\Disease-Modelling\\helper_scripts')

from luigi_helper import (
    # find_motion_changes,
    # read_metadata,
    plot_stimuli,
    short_plot_stimuli,
    load_bluelight_timeseries_from_results,
    # just_load_one_timeseries,
    count_motion_modes,
    get_frac_motion_modes,
    HIRES_COLS
    )

CUSTOM_STYLE = 'X:\\Bonnie\\Scripts\\Disease-Modelling\\gene_cards.mplstyle'
plt.style.use(CUSTOM_STYLE)

MODECOLNAMES=['frac_worms_fw', 'frac_worms_st', 'frac_worms_bw']

def align_bluelight_meta(metadata_df):
    # reshape the dataframe slightly so we only have one line per (plate,well)
    pre_df = metadata_df[metadata_df['imgstore_name'].str.contains('prestim')]
    blue_df = metadata_df[metadata_df['imgstore_name'].str.contains('bluelight')]
    post_df = metadata_df[metadata_df['imgstore_name'].str.contains('poststim')]

    cols_to_join = list(set(pre_df.columns) - set(['imgstore_name']))

    md_df = pd.merge(pre_df, blue_df,
                     how='outer',
                     on=cols_to_join,
                     suffixes=('_pre', '_blue'))
    md_df = pd.merge(md_df, post_df,
                     how='outer',
                     on=cols_to_join,
                     suffixes=('', '_post'))
    md_df.rename(columns={'imgstore_name': 'imgstore_name_post'},
                 inplace=True)
    
    return md_df

def my_sum_bootstrap(data, alpha=95):
    
    """
    Luigi's own bootstrapping with ci functions"""
    
    if isinstance(data, pd.core.series.Series):
        data_clean = data.values
        data_clean = data_clean[~np.isnan(data_clean)]
    else:
        data_clean = data[~np.isnan(data)]
    stafunction = np.sum
    n_samples = 1000
    funevals = np.ones(n_samples) * np.nan
    maxint = len(data_clean)
    sampling = np.random.randint(0, maxint, size=(n_samples, maxint))
    for evalc, sampling_ind in enumerate(sampling):
        funevals[evalc] = stafunction(data_clean[sampling_ind])
    pctiles = np.percentile(funevals, (50 - alpha/2, 50 + alpha/2))
    return tuple(pctiles)

def get_frac_motion_modes_with_ci(df, is_for_seaborn=False):
    """get_frac_motion_modes_with_ci
    divide number of worms in a motion mode by
    the total number of worms in that frame.
    Does *not* require that data have already been consolidated
    by a .groupby('timestamp')
    """
    # sum all n_worms across wells
    total_n_worms_in_frame = df.groupby(
        ['worm_gene', 'timestamp'], observed=True)['n_worms'].transform(sum)
    tmp = df.drop(columns='n_worms')
    # transform n_worms in frac_worms
    tmp = tmp.divide(total_n_worms_in_frame, axis=0)
    tmp.rename(lambda x: x.replace('n_worms_', 'frac_worms_'),
               axis='columns',
               inplace=True)
    if is_for_seaborn:
        # will use seaborn to get confidence intervals (estimator=sum),
        # don't need to calculate them here
        return tmp

    # implied else
    # now use bootstrap to get CIs
    out = tmp.groupby(['worm_gene', 'timestamp'],
                      observed=True).agg([np.sum, my_sum_bootstrap])

    def col_renamer(cname):
        cname_out = cname[0]+'_ci' if ('ci' in cname[1] or
                                       'bootstrap' in cname[1]) else cname[0]
        return cname_out
    out.columns = [col_renamer(c) for c in out.columns]

    return out


def make_feats_abs(timeseries_df,
                   feats_to_abs=['speed','relative_to_body_speed_midbody',
                                 'd_speed',
                                 'relative_to_neck_angular_velocity_head_tip']):
    # other feats to abs:
    # a few features only make sense if we know ventral/dorsal
    feats_to_abs.extend([f for f in timeseries_df.columns
                         if f.startswith(('path_curvature',
                                          'angular_velocity'))])
    for feat in feats_to_abs:
        timeseries_df['abs_' + feat] = timeseries_df[feat].abs()
    return timeseries_df

def select_ts_df(timeseries_df, strain_lut, CONTROL_STRAIN):
    
    from pandas.api.types import CategoricalDtype
    for k, v in strain_lut.items():
        if k != CONTROL_STRAIN:
            to_plot = [k, CONTROL_STRAIN]
            _plot_df = timeseries_df.query('motion_mode != 0 and @to_plot in worm_gene')
            _plot_df['worm_gene'] = _plot_df['worm_gene'].astype(
                CategoricalDtype(categories=to_plot, ordered=False)
                )
    
    return _plot_df


def plot_ts_features(plot_df, feature, strain_lut):
    """

    Parameters
    ----------
    plot_df : TYPE
        DESCRIPTION.
    feature : TYPE
        DESCRIPTION.
    strain_lut : TYPE
        DESCRIPTION.
    CONTROL_STRAIN : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from textwrap import wrap
    # with PdfPages(SAVETO / '{}_downsampled_feats.pdf'.format(ng), keep_empty=False) as pdf:
    fig, ax = plt.subplots(figsize=(7.5,5))
    sns.lineplot(x='time_binned_s',
                 y=feature,
                 # style='motion_mode',
                 hue='worm_gene',
                 hue_order=strain_lut.keys(),
                 data=plot_df,
                 # estimator=np.mean,
                 ci='sd',
                 # legend='full',
                 palette=strain_lut)
    plot_stimuli(ax=ax, units='s')
    ax.set_ylabel(ylabel='\n'.join(wrap(feature, 30)))
    plt.tight_layout()
                
    return

def plot_ts_features(plot_df, feature, strain_lut):
    """

    Parameters
    ----------
    plot_df : TYPE
        DESCRIPTION.
    feature : TYPE
        DESCRIPTION.
    strain_lut : TYPE
        DESCRIPTION.
    CONTROL_STRAIN : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from textwrap import wrap
    # with PdfPages(SAVETO / '{}_downsampled_feats.pdf'.format(ng), keep_empty=False) as pdf:
    fig, ax = plt.subplots(figsize=(7.5,5))
    sns.lineplot(x='time_binned_s',
                 y=feature,
                 # style='motion_mode',
                 hue='worm_gene',
                 hue_order=strain_lut.keys(),
                 data=plot_df,
                 # estimator=np.mean,
                 ci='sd',
                 # legend='full',
                 palette=strain_lut)
    plot_stimuli(ax=ax, units='s')
    ax.set_ylabel(ylabel='\n'.join(wrap(feature, 30)))
    plt.tight_layout()
                
    return

def plot_strains_ts(timeseries_df, strain_lut, CONTROL_STRAIN, features, SAVETO):
    
    plot_df = select_ts_df(timeseries_df, strain_lut, CONTROL_STRAIN)
    
    for f in features:
        plot_ts_features(plot_df, f, strain_lut)
        plt.savefig(SAVETO / '{}_downsampled_ts.png'.format(f), dpi=200)
        plt.close('all')
    return

#%% motion modes

def get_motion_modes(hires_df,
                     grouping=['worm_gene', 'timestamp'],
                     fps=25,
                     saveto=None):
    import time
    tic = time.time()
    motion_mode_by_well = count_motion_modes(hires_df)
    
    
    # aggregate data from all different wells, but keep strains separate
    motion_mode = motion_mode_by_well.groupby(grouping,
                                              observed=True).sum()

    # compute fraction of worms in each motion mode (works with 'worm_strain' too)
    frac_motion_mode_with_ci = get_frac_motion_modes_with_ci(
            motion_mode_by_well)
    for col in ['frac_worms_bw_ci', 'frac_worms_st_ci',
                'frac_worms_fw_ci', 'frac_worms_nan_ci']:
        frac_motion_mode_with_ci[col+'_lower'] = \
            frac_motion_mode_with_ci[col].apply(lambda x: x[0])
        frac_motion_mode_with_ci[col+'_upper'] = \
            frac_motion_mode_with_ci[col].apply(lambda x: x[1])
        frac_motion_mode_with_ci.drop(columns=col,
                                      inplace=True)
        
    frac_motion_mode_with_ci = frac_motion_mode_with_ci.reset_index()
    frac_motion_mode_with_ci['time_s'] = (frac_motion_mode_with_ci['timestamp']
                                          / fps)
    if saveto:
        frac_motion_mode_with_ci.to_hdf(saveto,
                                        'frac_motion_mode_with_ci',
                                        format='table')
    print('Time elapsed: {}s'.format(time.time()-tic))
    return motion_mode, frac_motion_mode_with_ci

def plot_frac_all_modes(df,
              strain,
              strain_lut,
              modecolnames=MODECOLNAMES):
              # ax=None,
              # **kwargs):
    styledict = {'frac_worms_fw': '-',
                  'frac_worms_st': ':',
                  'frac_worms_bw': '--',
                  'frac_worms_nan': '-'}
    
    plt.figure(figsize=(7.5,5))

    for col in modecolnames:
        plt.plot(df['time_s'],
                    df[col],
                    # ax=this_ax,
                    color=strain_lut[strain],
                    label=col, #'_nolegend_',
                    linestyle=styledict[col],
                    linewidth=2,
                    alpha=0.8)#),

        lower = df[col+'_ci_lower']
        upper = df[col+'_ci_upper']
        plt.fill_between(x=df['time_s'],
                             y1=lower.values,
                             y2=upper.values,
                             alpha=0.3,
                             facecolor=strain_lut[strain])

        plt.ylabel('fraction of worms')
        plt.xlabel('time, (s)')
        plt.title(strain)
        plt.ylim((0, 1))
        plt.legend(loc='upper right')
        plot_stimuli(units='s')
        plt.tight_layout()
    return


def plot_frac_all_modes_coloured_by_motion_mode(df,
              strain,
              strain_lut,
              modecolnames=MODECOLNAMES):
              # ax=None,
              # **kwargs):                
                  
    styledict = {'frac_worms_fw': '-',
                  'frac_worms_st': '-',
                  'frac_worms_bw': '-',
                  'frac_worms_nan': '-'}
    
    coldict = {'frac_worms_fw': ('tab:green'),
               'frac_worms_bw': ('tab:orange'),
               'frac_worms_st': ('tab:purple'),
               'frac_worms_nan': ('tab:gray')}  
    
    legenddict = {'frac_worms_fw': 'forwards',
               'frac_worms_bw':'backwards',
               'frac_worms_st': 'stationary', 
               'frac_worms_nan': 'undefined'}  
    
    
    plt.figure(figsize=(7.5,5))

    for col in modecolnames:
        plt.plot(df['time_s'],
                    df[col],
                    # ax=this_ax,
                    color=coldict[col],
                    label=legenddict[col], #'_nolegend_',
                    linestyle=styledict[col],
                    linewidth=2,
                    alpha=0.8)#),

        lower = df[col+'_ci_lower']
        upper = df[col+'_ci_upper']
        plt.fill_between(x=df['time_s'],
                             y1=lower.values,
                             y2=upper.values,
                             alpha=0.3,
                             facecolor=strain_lut[strain])

        plt.ylabel('fraction of worms')
        plt.xlabel('time, (s)')
        plt.title(strain)
        plt.ylim((0, 1))
        plt.legend(loc='upper right')
        plot_stimuli(units='s')
        plt.tight_layout()
    return

def plot_frac_by_mode(df,
                      strain_lut,
                      modecolname=MODECOLNAMES[0]):
                  # ax=None,
                  # **kwargs):
    # styledict = {'frac_worms_fw': '-',
    #               'frac_worms_st': ':',
    #               'frac_worms_bw': '--',
    #               'frac_worms_nan': '-'}

    plt.figure(figsize=(7.5,5))
    mode_dict = {'frac_worms_fw':'forward',
                 'frac_worms_bw':'backward',
                 'frac_worms_st': 'stationary'}
    
    
    for strain in list(strain_lut.keys()):
        plt.plot(df[df.worm_gene==strain]['time_s'],
                    df[df.worm_gene==strain][modecolname],
                    # ax=this_ax,
                    color=strain_lut[strain],
                    label=strain, #'_nolegend_',
                    # linestyle=styledict[col],
                    linewidth=2,
                    alpha=0.8)#),

        lower = df[df.worm_gene==strain][modecolname+'_ci_lower']
        upper = df[df.worm_gene==strain][modecolname+'_ci_upper']
        plt.fill_between(x=df[df.worm_gene==strain]['time_s'],
                             y1=lower.values,
                             y2=upper.values,
                             alpha=0.3,
                             facecolor=strain_lut[strain])

        plt.ylabel('fraction of worms')
        plt.xlabel('time, (s)')
        plt.title(mode_dict[modecolname])
        plt.ylim((0, 1))
        plt.legend(loc='upper right')
        plot_stimuli(units='s')
        plt.tight_layout()
    return

def short_plot_frac_by_mode(df,
                      strain_lut,
                      modecolname=MODECOLNAMES[0]):
                  # ax=None,
                  # **kwargs):
    # styledict = {'frac_worms_fw': '-',
    #               'frac_worms_st': ':',
    #               'frac_worms_bw': '--',
    #               'frac_worms_nan': '-'}

    plt.figure(figsize=(7.5,5))
    mode_dict = {'frac_worms_fw':'forward',
                 'frac_worms_bw':'backward',
                 'frac_worms_st': 'stationary'}
    
    
    for strain in list(strain_lut.keys()):
        plt.plot(df[df.worm_gene==strain]['time_s'],
                    df[df.worm_gene==strain][modecolname],
                    # ax=this_ax,
                    color=strain_lut[strain],
                    label=strain, #'_nolegend_',
                    # linestyle=styledict[col],
                    linewidth=2,
                    alpha=0.8)#),

        lower = df[df.worm_gene==strain][modecolname+'_ci_lower']
        upper = df[df.worm_gene==strain][modecolname+'_ci_upper']
        plt.fill_between(x=df[df.worm_gene==strain]['time_s'],
                             y1=lower.values,
                             y2=upper.values,
                             alpha=0.3,
                             facecolor=strain_lut[strain])

        plt.ylabel('fraction of worms')
        plt.xlabel('time, (s)')
        plt.title(mode_dict[modecolname])
        plt.ylim((0, 1))
        plt.legend(loc='upper right')
        short_plot_stimuli(units='s')
        plt.tight_layout()
    return



def plot_frac_by_mode_colour(df,
                      strain_lut,
                      modecolname=MODECOLNAMES[0]):
                  # ax=None,
                  # **kwargs):
    # styledict = {'frac_worms_fw': '-',
    #               'frac_worms_st': ':',
    #               'frac_worms_bw': '--',
    #               'frac_worms_nan': '-'}

    plt.figure(figsize=(7.5,5))
    mode_dict = {'frac_worms_fw':'forward',
                 'frac_worms_bw':'backward',
                 'frac_worms_st': 'stationary'}
    
    colordict = {'forward': 'green',
                'backward': 'purple',
                'stationary': 'orange',}
    
    for strain in list(strain_lut.keys()):
        plt.plot(df[df.worm_gene==strain]['time_s'],
                    df[df.worm_gene==strain][modecolname],
                    # ax=this_ax,
                    color=colordict,
                    label=strain, #'_nolegend_',
                    # linestyle=styledict[col],
                    linewidth=2,
                    alpha=0.8)#),

        lower = df[df.worm_gene==strain][modecolname+'_ci_lower']
        upper = df[df.worm_gene==strain][modecolname+'_ci_upper']
        plt.fill_between(x=df[df.worm_gene==strain]['time_s'],
                             y1=lower.values,
                             y2=upper.values,
                             alpha=0.3,
                             facecolor=strain_lut[strain])

        plt.ylabel('fraction of worms')
        plt.xlabel('time, (s)')
        plt.title(mode_dict[modecolname])
        plt.ylim((0, 1))
        plt.legend(loc='upper right')
        plot_stimuli(units='s')
        plt.tight_layout()
    return
    


def plot_group_frac(df,
              strain_lut,
              modecolnames=['frac_worms_fw', 'frac_worms_st', 'frac_worms_bw'],
              ax=None,
              **kwargs): #style_dict={'N2': '--','CB4856': '-'}
    """plot_frac
    plots modecolname of df with shaded errorbar
    example:
        plot_frac(frac_motion_mode_with_ci, 'frac_worms_fw', ax=ax)
    """
    if ax is None:
        ax = plt.gca()
    coldict = {'frac_worms_fw': 'tab:green',
                'frac_worms_st': 'tab:purple',
                'frac_worms_bw': 'tab:orange',
                'frac_worms_nan': 'tab:gray'}
    namesdict = {'frac_worms_fw': 'forwards',
                  'frac_worms_st': 'stationary',
                  'frac_worms_bw': 'backwards',
                  'frac_worms_nan': 'not defined'}
    
    if len(ax) != 1:
        assert(len(ax) == df['worm_gene'].nunique())

    for ii, (strain, df_g) in enumerate(df.groupby('worm_gene')):
        df_g = df_g.droplevel('worm_strain')
        if len(ax) != 1:
            this_ax = ax[ii]
        else:
            this_ax = ax
        
    plt.legend(frameon=False, loc='upper left')
    this_ax.get_legend().remove()
    for i, col in enumerate(modecolnames):
        xm, xM = this_ax.get_xlim()
        x = xm + 0.99 * (xM - xm)
        y = 0.95 - i * 0.05
        this_ax.text(x, y, namesdict[col], #linestyle=styledict[col],
                      fontweight='heavy',
                      horizontalalignment='right')
    return


