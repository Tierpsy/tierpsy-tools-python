#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:44:49 2020

@author: ibarlow

Helper functions for disease modelling
"""

"""
Created on Mon Nov 16 09:44:49 2021

@author: tobrien

Helper functions for disease modelling
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, '/Users/bonnie/Disease-Modelling/helper_scripts')

from helper import STIMULI_ORDER, BLUELIGHT_WINDOW_DICT
CUSTOM_STYLE = '/Users/bonnie/Disease-Modelling/gene_cards.mplstyle'
plt.style.use(CUSTOM_STYLE)

def plot_colormap(lut, orientation='vertical'):
    """

    Parameters
    ----------
    lut : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sns.set_style('dark')
    plt.style.use(CUSTOM_STYLE)

    from matplotlib import transforms
    if orientation == 'vertical':
        tr = transforms.Affine2D().rotate_deg(90)


        fig, ax = plt.subplots(1,1,
                               figsize=[4,5],
                             )
        ax.imshow([[v for v in lut.values()]],
                   transform=tr + ax.transData)
        ax.axes.set_ylim([-0.5, 0.5+len(lut.keys())-1])
        ax.axes.set_yticks(range(0,len(lut.keys())))
        ax.axes.set_yticklabels(lut.keys())

        ax.axes.set_xlim([0.5, -0.5])
        ax.set_xticklabels([])

    else:
        fig, ax = plt.subplots(1,1,
                               figsize=[5,2],
                             )
        ax.imshow([[v for v in lut.values()]])
        ax.axes.set_xticks(range(0, len(lut), 1))
        ax.axes.set_xticklabels(lut.keys(),
                                rotation=45,
                                fontdict={'fontsize':18,
                                          'weight':'bold'})
        ax.axes.set_yticklabels([])
        fig.tight_layout()

    return ax


def plot_cmap_text(lut, fsize=60):
    """
    Plot text as colour

    Parameters
    ----------
    lut : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import matplotlib.gridspec as gridspec
    import matplotlib.patheffects as PathEffects


    plt.figure(figsize = [5,
                          len(lut)*2.5])

    gs1 = gridspec.GridSpec(len(lut),
                           1)
    gs1.update(wspace=-0.01, hspace=0) # set the spacing between axes.
    # fig, axes = plt.subplots(len(lut),
    #                        1,
    #                            figsize=[5,
    #                                     len(lut)*2.5],
    #                          gridspec_kw = {'wspace':0, 'hspace':0},
    #                          constrained_layout=True)
    # fig.subplots_adjust(hspace=0.05)
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)

    for c, (k,v) in enumerate(lut.items()):
        ax1 = plt.subplot(gs1[c])
        ax1.text(y=0.5,
                x=0.5,
                s=k,
                verticalalignment='center',
                horizontalalignment='center',
                fontdict={'fontsize':fsize,
                          'weight':'bold',
                          'color':v,
                          'style':'italic' if k!='N2' else 'normal'},
                path_effects=[PathEffects.withStroke(linewidth=1, foreground='k')])
        ax1.axis("off") 

    return


# def plot_colormaps(strain_lut, stim_lut, saveto):
#     """


#     Parameters
#     ----------
#     strain_lut : TYPE
#         DESCRIPTION.
#     stim_lut : TYPE
#         DESCRIPTION.
#     saveto : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     from matplotlib import transforms
#     tr = transforms.Affine2D().rotate_deg(90)
#     sns.set_style('dark')
#     plt.style.use(CUSTOM_STYLE)

#     fig, ax = plt.subplots(2,1,
#                            figsize=[4,10],
#                            gridspec_kw={'height_ratios': [1,1]})
#     # ax=ax.flatten()
#     for c, (axis, lut) in enumerate([(ax[0], strain_lut), (ax[1], stim_lut)]):
#         axis.imshow([[v for v in lut.values()]],
#                     transform=tr + axis.transData)
#         axis.axes.set_ylim([-0.5, 0.5+len(lut.keys())-1])
#         axis.axes.set_yticks(range(0,len(lut.keys())))
#         axis.axes.set_yticklabels(lut.keys())

#         axis.axes.set_xlim([0.5, -0.5])
#         axis.set_xticklabels([])

#     # fig.tight_layout()
#     if saveto==None:
#         return
#     else:
#         fig.savefig(saveto / 'colormaps.png')

#     return

def make_clustermaps(featZ, meta, featsets, strain_lut, feat_lut, 
                     group_vars=['worm_gene','date_yyyymmdd'], row_color = 'worm_gene',
                     saveto=None):
    """
    Parameters
    ----------
    featZ : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    featsets : TYPE
        DESCRIPTION.
    strain_lut : TYPE
        DESCRIPTION.
    feat_lut : TYPE
        DESCRIPTION.
    saveto : TYPE
        DESCRIPTION.
    group_vars : TYPE, optional
        DESCRIPTION. The default is ['worm_gene','imaging_date_yyyymmdd'].

    Returns
    -------
    clustered_features : TYPE
        DESCRIPTION.

    """
    plt.style.use(CUSTOM_STYLE)

    featZ_grouped = pd.concat([featZ,
                               meta],
                              axis=1
                              ).groupby(group_vars).mean()
    featZ_grouped.reset_index(inplace=True)

    row_colors = featZ_grouped[row_color].map(strain_lut)

    # make clustermaps
    clustered_features = {}
    sns.set(font_scale=1.2)
    for stim, fset in featsets.items():
        col_colors = featZ_grouped[fset].columns.map(feat_lut)   #This clustermap is coloured using the feature look up table
        plt.figure(figsize=[7.5,5])
        cg = sns.clustermap(featZ_grouped[fset],
                        row_colors=row_colors,
                        col_colors=col_colors,
                        vmin=-2,
                        vmax=2,
                        yticklabels=False)
        cg.ax_heatmap.axes.set_xticklabels([])
        cg.ax_heatmap.axes.set_xlabel('')
        # cg.ax_heatmap.axes.set_yticklabels([])
        if saveto!=None:
            # cg.savefig(Path(saveto) / '{}_clustermap_dpi1000_resolution.png'.format(stim), dpi=1000)
            cg.savefig(Path(saveto) / '{}_clustermap.png'.format(stim), dpi=300)
            # cg.savefig(Path(saveto) / '{}_clustermap.svg'.format(stim))


        clustered_features[stim] = np.array(featsets[stim])[cg.dendrogram_col.reordered_ind]
        plt.close('all')

    return clustered_features

def make_paper_clustermaps(featZ, meta, featsets, strain_lut, feat_lut, group_vars=['worm_gene','date_yyyymmdd'], saveto=None):
    """
    Parameters
    ----------
    featZ : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    featsets : TYPE
        DESCRIPTION.
    strain_lut : TYPE
        DESCRIPTION.
    feat_lut : TYPE
        DESCRIPTION.
    saveto : TYPE
        DESCRIPTION.
    group_vars : TYPE, optional
        DESCRIPTION. The default is ['worm_gene','imaging_date_yyyymmdd'].

    Returns
    -------
    clustered_features : TYPE
        DESCRIPTION.

    """
    plt.style.use(CUSTOM_STYLE)

    featZ_grouped = pd.concat([featZ,
                               meta],
                              axis=1
                              ).groupby(group_vars).mean()
    featZ_grouped.reset_index(inplace=True)

    row_colors = featZ_grouped['worm_gene'].map(strain_lut)

    # make clustermaps
    clustered_features = {}
    sns.set(font_scale=1.2)
    for stim, fset in featsets.items():
        col_colors = featZ_grouped[fset].columns.map(feat_lut)   #This clustermap is coloured using the feature look up table
        plt.figure(figsize=[7.5,5])
        cg = sns.clustermap(featZ_grouped[fset],
                        row_colors=None,
                        col_colors=col_colors,
                        vmin=-2,
                        vmax=2,
                        yticklabels=True)
        cg.ax_heatmap.axes.set_xticklabels([])
        cg.ax_heatmap.axes.set_yticklabels(featZ_grouped['worm_gene'], fontsize=10)
        if saveto!=None:
            # cg.savefig(Path(saveto) / '{}_clustermap_dpi1000_resolution.png'.format(stim), dpi=1000)
            cg.savefig(Path(saveto) / '{}_clustermap_with_names.png'.format(stim), dpi=300)
            # cg.savefig(Path(saveto) / '{}_clustermap.svg'.format(stim))


        clustered_features[stim] = np.array(featsets[stim])[cg.dendrogram_col.reordered_ind]
        plt.close('all')

    return clustered_features

def make_barcode(heatmap_df, selected_feats, cm=['inferno', 'inferno', 'Greys', 'Pastel1'], vmin_max = [(-2,2), (-2,2), (0,4), (1,3)]):  #Change 3rd set of parameters to change p-val scale bar
    """

    Parameters
    ----------
    heatmap_df : TYPE
        DESCRIPTION.
    selected_feats : TYPE
        DESCRIPTION.
    cm : TYPE, optional
        DESCRIPTION. The default is ['inferno', 'inferno', 'gray', 'Pastel1'].
    vmin_max : TYPE, optional
        DESCRIPTION. The default is [(-2,2), (-2,2), (-20, 0), (1,3)].

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    from matplotlib.gridspec import GridSpec
    sns.set_style('ticks')
    plt.style.use(CUSTOM_STYLE)

    fig_ratios = list(np.ones(heatmap_df.shape[0]))
    fig_ratios = [i*3 if c<(len(fig_ratios)-1) else i for c,i in enumerate(fig_ratios)]
    
    f = plt.figure(figsize= (24,3))
    gs = GridSpec(heatmap_df.shape[0], 1,
                  wspace=0,
                  hspace=0,
                  height_ratios=fig_ratios)
    
    cbar_axes = [f.add_axes([.89, .3, .02, .4]), [],
               f.add_axes([.935, .3, .02, .4]),[]]

    
    for n, ((ix,r), c, v) in enumerate(zip(heatmap_df.iterrows(), cm, vmin_max)):
        axis = f.add_subplot(gs[n])
        
        if ix != 'stim_type' and n<3:
            sns.heatmap(r.to_frame().transpose().astype(float),
                        yticklabels=[ix],
                        xticklabels=[],
                        ax=axis,
                        cmap=c,
                        cbar=n==0 or n==2, #only plots colorbar for first plot
                        cbar_ax=cbar_axes[n],#None if n else cbar_axes[n],
                        vmin=v[0],
                        vmax=v[1])
            axis.set_yticklabels(labels=[ix],
                                 rotation=0,
                                  fontsize=20
                                 )
        elif ix != 'stim_type':
            sns.heatmap(r.to_frame().transpose().astype(float),
                        yticklabels=[ix],
                        xticklabels=[],
                        ax=axis,
                        cmap=c,
                        cbar=False,
                        vmin=v[0],
                        vmax=v[1])
            axis.set_yticklabels(labels=[ix],
                                 rotation=0,
                                  fontsize=20
                                 )
        else:
            c = sns.color_palette('Pastel1',3)
            sns.heatmap(r.to_frame().transpose(),
                    yticklabels=[ix],
                    xticklabels=[],
                    ax=axis,
                    cmap=c,
                    cbar=n==0,
                    cbar_ax=None if n else cbar_axes[n],
                    vmin=v[0],
                    vmax=v[1])
            axis.set_yticklabels(labels=[ix],
                                 rotation=0,
                                  fontsize=20
                                 )
        cbar_axes[0].set_yticklabels(labels=cbar_axes[0].get_yticklabels())#, fontdict=font_settings)
        cbar_axes[2].set_yticklabels(labels=['>0.05',
                                             np.power(10,-vmin_max[2][1]/2),
                                             np.power(10,-float(vmin_max[2][1]))
                                             ]
                                     ) 
        # f.tight_layout()
        f.tight_layout(rect=[0, 0, 0.89, 1], w_pad=0.5)

    for sf in selected_feats:
        try:
            axis.text(heatmap_df.columns.get_loc(sf), 1, '*', fontsize=20)
        except KeyError:
            print('{} not in featureset'.format(sf))
    return f

def make_heatmap_df(fset, featZ, meta, p_vals=None, groupby = 'worm_gene'):
    """

    Parameters
    ----------
    fset : TYPE
        DESCRIPTION.
    featZ : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    p_vals : TYPE
        DESCRIPTION.
    cm : TYPE, optional
        DESCRIPTION. The default is 'worm_gene'.

    Returns
    -------
    heatmap_df : TYPE
        DESCRIPTION.

    """

    heatmap_df = [pd.concat([featZ,
                        meta],
                       axis=1
                       ).groupby(groupby).mean()[fset]]
    if p_vals is not None:
        try:
            heatmap_df.append(-np.log10(p_vals[fset]))
        except TypeError:
            print('p values not logged')
            heatmap_df.append(p_vals[fset])

    _stim = pd.DataFrame(data=[i.split('_')[-1] for i in fset],
                         columns=['stim_type'])
    _stim['stim_type'] = _stim['stim_type'].map(STIMULI_ORDER)
    _stim = _stim.transpose()
    _stim.rename(columns={c:v for c,v in enumerate(fset)}, inplace=True)
    heatmap_df.append(_stim)

    heatmap_df = pd.concat(heatmap_df)
    return heatmap_df

def clustered_barcodes(clustered_feats_dict, selected_feats, featZ, meta, p_vals, saveto):
    """

    Parameters
    ----------
    clustered_feats_dict : TYPE
        DESCRIPTION.
    selected_feats : TYPE
        DESCRIPTION.
    featZ : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    p_vals : TYPE
        DESCRIPTION.
    saveto : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plt.style.use(CUSTOM_STYLE)
    for stim, fset in clustered_feats_dict.items():

        if stim !='all':
            missing_feats = list(set(fset).symmetric_difference([f for f in featZ.columns if stim in f]))
        else:
            missing_feats = list(set(fset).symmetric_difference(featZ.columns))

        if len(missing_feats)>0:
            for i in missing_feats:
                try:
                    if isinstance(fset, np.ndarray):
                        fset = fset.tolist()
                    fset.remove(i) 
                except ValueError:
                    print('{} not in {} feature set'.format(i, stim))
        heatmap_df = make_heatmap_df(fset, featZ, meta, p_vals)

        f = make_barcode(heatmap_df, selected_feats)
        f.savefig(saveto / '{}_heatmap.png'.format(stim))
    return

def feature_box_plots(feature, feat_df, meta_df, strain_lut, show_raw_data=True, bhP_values_df=None, add_stats=True):
    """

    Parameters
    ----------
    feature : TYPE
        DESCRIPTION.
    feat_df : TYPE
        DESCRIPTION.
    meta_df : TYPE
        DESCRIPTION.
    bhP_values_df : TYPE
        DESCRIPTION.
    strain_lut : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from statannot import add_stat_annotation
    from decimal import Decimal
    label_format = '{0:.4g}'
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    plt.tight_layout()

    plt.figure(figsize=(5,10))
    # plt.figure(figsize=(15,10))
    ax = sns.boxplot(y=feature,
                x='worm_gene',
                data=pd.concat([feat_df, meta_df],
                               axis=1),
                order=strain_lut.keys(),                
                # order=['N2', 'hlb-1', '50uM_carbachol_N2_4h', '50uM_carbachol_hlb-1_4h', '10uM_carbachol_N2_4h', '10uM_carbachol_hlb-1_4h', '1uM_carbachol_N2_4h', '1uM_carbachol_hlb-1_4h'],
                palette=strain_lut.values(),
                showfliers=False)
    plt.tight_layout()
    if show_raw_data=='date':
        sns.swarmplot(y=feature,
                x='worm_gene',
                data=pd.concat([feat_df, meta_df],
                               axis=1),
                order=strain_lut.keys(),
                # order=['N2', 'hlb-1', '50uM_carbachol_N2_4h', '50uM_carbachol_hlb-1_4h', '10uM_carbachol_N2_4h', '10uM_carbachol_hlb-1_4h', '1uM_carbachol_N2_4h', '1uM_carbachol_hlb-1_4h'],
                hue='date_yyyymmdd',
                palette='Greys',
                alpha=0.6)
    ax.set_ylabel(fontsize=22, ylabel=feature)
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])#, fontsize=16) #labels = ax.get_yticks(),
    ax.set_xlabel('')
    ax.set_xticklabels(labels = strain_lut.keys(), rotation=90)    
    # ax.legend(title='date_yyyy-mm-dd')
    # plt.legend(loc='lower right')
    plt.legend(loc='upper right')
    plt.legend(title = 'date_yyyy-mm-dd', title_fontsize = 14,fontsize = 14, 
                bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.tight_layout()
    
    # if show_raw_data=='drug_conc':
    #     sns.swarmplot(y=feature,
    #             x='worm_gene',
    #             data=pd.concat([feat_df, meta_df],
    #                            axis=1),
    #             order=strain_lut.keys(),
    #             # order=['N2', 'hlb-1', '50uM_carbachol_N2_4h', '50uM_carbachol_hlb-1_4h', '10uM_carbachol_N2_4h', '10uM_carbachol_hlb-1_4h', '1uM_carbachol_N2_4h', '1uM_carbachol_hlb-1_4h'],
    #             hue='imaging_plate_drug_concentration_uM',
    #             palette='Greys',
    #             alpha=0.6)
    # ax.set_ylabel(fontsize=22, ylabel=feature)
    # ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])#, fontsize=16) #labels = ax.get_yticks(),
    # ax.set_xlabel('')
    # ax.set_xticklabels(labels = strain_lut.keys())
    # plt.legend(loc='lower right')
    # plt.tight_layout()

    if add_stats:
                
        if bhP_values_df is not None:
            add_stat_annotation(ax,
                                data=pd.concat([feat_df,
                                                meta_df], axis=1),
                                x='worm_gene',
                                y=feature,
                                order=strain_lut.keys(),
                                box_pairs=[strain_lut.keys()],
                                perform_stat_test=False,
                                pvalues=[bhP_values_df[feature].values[0]], #pvalues=[bhP_values_df[feature].values[0]],
                                test=None,
                                # text_format='star',
                                loc='outside',
                                verbose=2,
                                text_annot_custom=['p={:.3E}'.format(round(bhP_values_df[feature].values[0],100))], #:.4f
                                fontsize=20,
                                )
            plt.tight_layout()

        else:
            import itertools
            add_stat_annotation(ax,
                                data=pd.concat([feat_df,
                                                meta_df], axis=1),
                                x='worm_gene',
                                y=feature,
                                order=strain_lut.keys(),
                                box_pairs=list(itertools.combinations(strain_lut.keys(),2)),
                                perform_stat_test=True,
                                # pvalues=[bhP_values_df[feature].values[0]],
                                test='Kruskal',
                                text_format='full',
                                # show_test_name=False,
                                # test_short_name=None,
                                loc='outside',
                                verbose=2,
                                # text_annot_custom=['p={:.3E}'.format(round(bhP_values_df[feature].values[0],100)), #:.4f
                                fontsize=14,
                                line_offset=0.05,
                                line_offset_to_box=0.05
                                )
            plt.tight_layout()

    if len(strain_lut) > 2:
        plt.xticks(rotation=90)
        plt.tight_layout()

    return
    
def clipped_feature_box_plots(feature, feat_df, meta_df, strain_lut, top_clip, bottom_clip, show_raw_data=True, bhP_values_df=None, add_stats=True):
    """

    Parameters
    ----------
    feature : TYPE
        DESCRIPTION.
    feat_df : TYPE
        DESCRIPTION.
    meta_df : TYPE
        DESCRIPTION.
    bhP_values_df : TYPE
        DESCRIPTION.
    strain_lut : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from statannot import add_stat_annotation
    from decimal import Decimal
    label_format = '{0:.4g}'
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')

    plt.figure(figsize=(5,10))
    ax = sns.boxplot(y=feature,
                x='worm_gene',
                data=pd.concat([feat_df, meta_df],
                               axis=1),
                order=strain_lut.keys(),
                palette=strain_lut.values(),
                showfliers=False)
    # yclip = ax.get_ylim()[1]
    # yclip=ax.set_ylim(top=1)
    _, yclip = ax.set_ylim(
                            top=top_clip,
                            bottom=bottom_clip
                            )
    if show_raw_data:
        full_df = pd.concat([feat_df, meta_df],
                               axis=1)
    sns.swarmplot(y=full_df[feature].clip(lower=None, upper=yclip),
                x=full_df['worm_gene'],
                order=strain_lut.keys(),
                hue=full_df['date_yyyymmdd'],
                palette='Greys',
                alpha=0.6)
    ax.set_ylabel(fontsize=22, ylabel=feature)
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])#, fontsize=16) #labels = ax.get_yticks(),
    ax.set_xlabel('')
    ax.set_xticklabels(labels = strain_lut.keys())
    if add_stats:
                
        if bhP_values_df is not None:
            add_stat_annotation(ax,
                                data=pd.concat([feat_df,
                                                meta_df], axis=1),
                                x='worm_gene',
                                y=feature,
                                order=strain_lut.keys(),
                                box_pairs=[strain_lut.keys()],
                                perform_stat_test=False,
                                pvalues=[bhP_values_df[feature].values[0]], #pvalues=[bhP_values_df[feature].values[0]],
                                test=None,
                                # text_format='star',
                                loc='outside',
                                verbose=2,
                                text_annot_custom=['p={:.3E}'.format(round(bhP_values_df[feature].values[0],100))], #:.4f
                                fontsize=20,
                                )
            plt.tight_layout()

        else:
            import itertools
            add_stat_annotation(ax,
                                data=pd.concat([feat_df,
                                                meta_df], axis=1),
                                x='worm_gene',
                                y=feature,
                                order=strain_lut.keys(),
                                box_pairs=list(itertools.combinations(strain_lut.keys(),2)),
                                perform_stat_test=True,
                                # pvalues=[bhP_values_df[feature].values[0]],
                                test='Kruskal',
                                text_format='full',
                                # show_test_name=False,
                                # test_short_name=None,
                                loc='outside',
                                verbose=2,
                                # text_annot_custom=['p={:.3E}'.format(round(bhP_values_df[feature].values[0],100)), #:.4f
                                fontsize=14,
                                line_offset=0.05,
                                line_offset_to_box=0.05
                                )
            plt.tight_layout()

    if len(strain_lut) > 2:
        plt.xticks(rotation=90)
        plt.tight_layout()

    return
def window_errorbar_plots(feature, feat, meta, cmap_lut, plot_legend=False):
    """


    Parameters
    ----------
    feature : TYPE
        DESCRIPTION.
    feat : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    cmap_lut : TYPE
        DESCRIPTION.
    plot_legend : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    import matplotlib.patches as patches
    from textwrap import wrap
    label_format = '{0:.4g}'
    plt.style.use(CUSTOM_STYLE)

    n_stim = meta.stim_number.unique().shape[0]

    _window_grouped = pd.concat([feat,
                                 meta],
                                axis=1).groupby(['window_sec',
                                                     'worm_gene'])[feature].describe().reset_index()

    fig, ax = plt.subplots(figsize=[(n_stim*2)+2,6])
    for g in meta.worm_gene.unique():

        xs = _window_grouped.query('@g in worm_gene')['window_sec']
        ys = _window_grouped.query('@g in worm_gene')['mean']
        yerr = _window_grouped.query('@g in worm_gene')['mean'] / (_window_grouped.query('@g in worm_gene')['count'])**0.5 #['std'] / _windw

        plt.errorbar(xs,
                     ys,
                     yerr,
                     fmt='-o',
                     color=cmap_lut[g],
                     alpha=0.8,
                     linewidth=2,
                     axes=ax,
                     label=g)
        if plot_legend:
            plt.legend()

    ax.set_ylabel(fontsize=18,
                  ylabel='\n'.join(wrap(feature, 25)))
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()],
                       fontsize=16
                       )
    ax.set_xlabel(fontsize=14,
                  xlabel='window')
    # plt.xticks(ticks=[x[0] for x in BLUELIGHT_WINDOW_DICT.values()],
    #            labels=[x[1] for x in BLUELIGHT_WINDOW_DICT.values()])
    ax.set_xticks(ticks=xs)
    ax.set_xticklabels(labels=[x[1] for x in BLUELIGHT_WINDOW_DICT.values() if x[2] in meta.stim_number.unique()],
                        fontsize=12,
                        rotation=45)
    y_min = ax.axes.get_ylim()[0]
    y_max = ax.axes.get_ylim()[1]
    if y_min<y_max:
        rects = (patches.Rectangle((60, y_min), 10, abs(y_min-y_max),
                                   facecolor='tab:blue',
                                   alpha=0.3),
                 patches.Rectangle((160, y_min), 10, abs(y_min-y_max),
                                   facecolor='tab:blue',
                                   alpha=0.3),
                 patches.Rectangle((260, y_min), 10, abs(y_min-y_max),
                                   facecolor='tab:blue',
                                   alpha=0.3))

    [ax.add_patch(r) for r in rects]

    plt.tight_layout()
    return


def average_feature_box_plots(feature, feat_df, meta_df, strain_lut, show_raw_data=True, bhP_values_df=None, add_stats=True):
    """

    Parameters
    ----------
    feature : TYPE
        DESCRIPTION.
    feat_df : TYPE
        DESCRIPTION.
    meta_df : TYPE
        DESCRIPTION.
    bhP_values_df : TYPE
        DESCRIPTION.
    strain_lut : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from statannot import add_stat_annotation
    from decimal import Decimal
    label_format = '{0:.4g}'
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    plt.tight_layout()

    plt.figure(figsize=(5,10))
    # plt.figure(figsize=(15,10))
    ax = sns.boxplot(y=feature,
                x='worm_gene',
                data=pd.concat([feat_df, meta_df],
                               axis=1),
                order=strain_lut.keys(),
                # order=['N2', 'hlb-1', '50uM_carbachol_N2_4h', '50uM_carbachol_hlb-1_4h', '10uM_carbachol_N2_4h', '10uM_carbachol_hlb-1_4h', '1uM_carbachol_N2_4h', '1uM_carbachol_hlb-1_4h'],
                palette=strain_lut.values(),
                showfliers=False)
    plt.tight_layout()
    
    if show_raw_data=='date':   
        group_vars =['worm_gene', 'date_yyyymmdd']
        av_data=pd.concat([feat_df,
                        meta_df],
                        axis=1
                        ).groupby(by=group_vars).mean().reset_index() 
        sns.swarmplot(
                y=feature,
                x='worm_gene',
                data=av_data,
                order=strain_lut.keys(),
                # order=['N2', 'hlb-1', '50uM_carbachol_N2_4h', '50uM_carbachol_hlb-1_4h', '10uM_carbachol_N2_4h', '10uM_carbachol_hlb-1_4h', '1uM_carbachol_N2_4h', '1uM_carbachol_hlb-1_4h'],
                hue='date_yyyymmdd',
                palette='colorblind',
                alpha=0.6,)
    ax.set_ylabel(fontsize=22, ylabel=feature)
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])#, fontsize=16) #labels = ax.get_yticks(),
    ax.set_xlabel('')
    ax.set_xticklabels(labels = strain_lut.keys())
    # plt.legend([],[], frameon=False)
    plt.legend(title= 'date_yyyymmdd', loc='upper right')
    plt.tight_layout()

    if show_raw_data=='drug_conc':   
        group_vars =['worm_gene', 'imaging_plate_drug_concentration_uM']
        av_data=pd.concat([feat_df,
                        meta_df],
                        axis=1
                        ).groupby(by=group_vars).mean().reset_index() 
        sns.swarmplot(
                y=feature,
                x='worm_gene',
                data=av_data,
                order=strain_lut.keys(),
                # order=['N2', 'hlb-1', '50uM_carbachol_N2_4h', '50uM_carbachol_hlb-1_4h', '10uM_carbachol_N2_4h', '10uM_carbachol_hlb-1_4h', '1uM_carbachol_N2_4h', '1uM_carbachol_hlb-1_4h'],
                hue='imaging_plate_drug_concentration_uM',
                palette='colorblind',
                alpha=0.6)
    ax.set_ylabel(fontsize=22, ylabel=feature)
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])#, fontsize=16) #labels = ax.get_yticks(),
    ax.set_xlabel('')
    ax.set_xticklabels(labels = strain_lut.keys())
    # plt.legend([],[], frameon=False)
    plt.legend(loc='upper right')
    plt.tight_layout()
  
    
    if add_stats:
        
        if bhP_values_df is not None:
            add_stat_annotation(ax,
                                data=pd.concat([feat_df,
                                                meta_df], axis=1),
                                x='worm_gene',
                                y=feature,
                                order=strain_lut.keys(),
                                box_pairs=[strain_lut.keys()],
                                perform_stat_test=False,
                                pvalues=[bhP_values_df[feature].values[0]], #pvalues=[bhP_values_df[feature].values[0]],
                                test=None,
                                # text_format='star',
                                loc='outside',
                                verbose=2,
                                text_annot_custom=['p={:.3E}'.format(round(bhP_values_df[feature].values[0],100))], #:.4f
                                fontsize=20,
                                )
            plt.tight_layout()
            

        else:
            import itertools
            add_stat_annotation(ax,
                                data=pd.concat([feat_df,
                                                meta_df], axis=1),
                                x='worm_gene',
                                y=feature,
                                order=strain_lut.keys(),
                                box_pairs=list(itertools.combinations(strain_lut.keys(),2)),
                                perform_stat_test=True,
                                # pvalues=[bhP_values_df[feature].values[0]],
                                test='Kruskal',
                                text_format='full',
                                # show_test_name=False,
                                # test_short_name=None,
                                loc='outside',
                                verbose=2,
                                # text_annot_custom=['p={:.3E}'.format(round(bhP_values_df[feature].values[0],100)), #:.4f
                                fontsize=14,
                                line_offset=0.05,
                                line_offset_to_box=0.05
                                )
            plt.tight_layout()

    if len(strain_lut) > 2:
        plt.xticks(rotation=90)
        plt.tight_layout()

    return

def average_feat_swarm(feature, feat_df, meta_df, strain_lut, show_raw_data=True, bhP_values_df=None, add_stats=True):
    """

    Parameters
    ----------
    feature : TYPE
        DESCRIPTION.
    feat_df : TYPE
        DESCRIPTION.
    meta_df : TYPE
        DESCRIPTION.
    bhP_values_df : TYPE
        DESCRIPTION.
    strain_lut : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from statannot import add_stat_annotation
    from decimal import Decimal
    label_format = '{0:.4g}'
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    plt.tight_layout()

    plt.figure(figsize=(5,10))

    
    if show_raw_data=='date':   
        group_vars =['worm_gene', 'date_yyyymmdd']
        av_data=pd.concat([feat_df,
                        meta_df],
                        axis=1
                        ).groupby(by=group_vars).mean().reset_index() 
        ax = sns.swarmplot(
                    y=feature,
                    x='worm_gene',
                    data=av_data,
                    order=strain_lut.keys(),
                    hue='date_yyyymmdd',
                    palette='bright',
                    alpha=0.6)
        ax.set_ylabel(fontsize=22, ylabel=feature)
        ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])#, fontsize=16) #labels = ax.get_yticks(),
        ax.set_xlabel('')
        ax.set_xticklabels(labels = strain_lut.keys(), rotation=90)
        # ax.xticks(rotation=90)
        plt.tight_layout()

    if show_raw_data=='drug_conc':   
        group_vars =['worm_gene', 'imaging_plate_drug_concentration_uM']
        av_data=pd.concat([feat_df,
                        meta_df],
                        axis=1
                        ).groupby(by=group_vars).mean().reset_index() 
        ax = sns.swarmplot(
                y=feature,
                x='worm_gene',
                data=av_data,
                order=strain_lut.keys(),
                hue='imaging_plate_drug_concentration_uM',
                palette='bright',
                alpha=0.6)
        ax.set_ylabel(fontsize=22, ylabel=feature)
        ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])#, fontsize=16) #labels = ax.get_yticks(),
        ax.set_xlabel('')
        ax.set_xticklabels(labels = strain_lut.keys(),rotation=90)
        # ax.xticks(rotation=90)
        plt.tight_layout()
  
    
    if add_stats:
        
        if bhP_values_df is not None:
            add_stat_annotation(ax,
                                data=pd.concat([feat_df,
                                                meta_df], axis=1),
                                x='worm_gene',
                                y=feature,
                                order=strain_lut.keys(),
                                box_pairs=[strain_lut.keys()],
                                perform_stat_test=False,
                                pvalues=[bhP_values_df[feature].values[0]], #pvalues=[bhP_values_df[feature].values[0]],
                                test=None,
                                # text_format='star',
                                loc='outside',
                                verbose=2,
                                text_annot_custom=['p={:.3E}'.format(round(bhP_values_df[feature].values[0],100))], #:.4f
                                fontsize=20,
                                )
            plt.tight_layout()
