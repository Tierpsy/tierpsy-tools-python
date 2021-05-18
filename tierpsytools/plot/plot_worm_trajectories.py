#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:26:37 2020

@author: em812
"""
pos96wellplate = {
    'A1': [0,0], 'A2': [0,1], 'A3': [0,2], 'A4': [0,3],
    'B1': [1,0], 'B2': [1,1], 'B3': [1,2], 'B4': [1,3],
    'C1': [2,0], 'C2': [2,1], 'C3': [2,2], 'C4': [2,3],
    'D1': [3,0], 'D2': [3,1], 'D3': [3,2], 'D4': [3,3],

    'A5': [0,0], 'A6': [0,1], 'A7': [0,2], 'A8': [0,3],
    'B5': [1,0], 'B6': [1,1], 'B7': [1,2], 'B8': [1,3],
    'C5': [2,0], 'C6': [2,1], 'C7': [2,2], 'C8': [2,3],
    'D5': [3,0], 'D6': [3,1], 'D7': [3,2], 'D8': [3,3],

    'A9': [0,0], 'A10': [0,1], 'A11': [0,2], 'A12': [0,3],
    'B9': [1,0], 'B10': [1,1], 'B11': [1,2], 'B12': [1,3],
    'C9': [2,0], 'C10': [2,1], 'C11': [2,2], 'C12': [2,3],
    'D9': [3,0], 'D10': [3,1], 'D11': [3,2], 'D12': [3,3],

    'E1': [0,0], 'E2': [0,1], 'E3': [0,2], 'E4': [0,3],
    'F1': [1,0], 'F2': [1,1], 'F3': [1,2], 'F4': [1,3],
    'G1': [2,0], 'G2': [2,1], 'G3': [2,2], 'G4': [2,3],
    'H1': [3,0], 'H2': [3,1], 'H3': [3,2], 'H4': [3,3],

    'E5': [0,0], 'E6': [0,1], 'E7': [0,2], 'E8': [0,3],
    'F5': [1,0], 'F6': [1,1], 'F7': [1,2], 'F8': [1,3],
    'G5': [2,0], 'G6': [2,1], 'G7': [2,2], 'G8': [2,3],
    'H5': [3,0], 'H6': [3,1], 'H7': [3,2], 'H8': [3,3],

    'E9': [0,0], 'E10': [0,1], 'E11': [0,2], 'E12': [0,3],
    'F9': [1,0], 'F10': [1,1], 'F11': [1,2], 'F12': [1,3],
    'G9': [2,0], 'G10': [2,1], 'G11': [2,2], 'G12': [2,3],
    'H9': [3,0], 'H10': [3,1], 'H11': [3,2], 'H12': [3,3],
     }

import matplotlib.pyplot as plt
import pandas as pd
import pdb

def plot_well_trajectories(
        xycoord, worm_ids, subsampling_rate=1, title=None,
        xlim=None, ylim=None, saveto=None):
    """
    Plots the trajectories of one well.

    Parameters
    ----------
    xycoord : dataframe shape = ( sum(trajectories_lengths) , 2)
        A dataframe with 'coord_x' and 'coord_y' columns containing all the
        trajectories timeseries of the well.
    worm_ids : array-like shape = (sum(trajectories_lengths), )
        Contains the well id for each row of xycoord.
    subsampling_rate : int, optional
        Subsampling rate for plotting purposes. The default is 1.
    title : string, optional
        The title of the plot. The default is None.
    xlim : list len=2, optional
        The limits of the plot along the x axis. The default is None.
    ylim : list len=2, optional
        The limits of the plot along the y axis. The default is None.
    saveto : path, optional
        The full path to the file where the figure will be saved. If None, the
        figure will not be saved and will be kept open. The default is None.

    Returns
    -------
    None.

    """
    from numpy import unique
    import textwrap

    n_traj = unique(worm_ids).shape[0]
    if title is None:
        title = '{} trajectories'.format(n_traj)
    else:
        title = textwrap.fill(title, 50)
    xycoord = pd.DataFrame(xycoord)

    fig, ax = plt.subplots()
    ax.set_title(title)
    xycoord.groupby(by=worm_ids).apply(
        _plot_trajectory, axes=ax, subsampling_rate=subsampling_rate
        )
    if xlim is None and ylim is None:
        ax.axis('equal')
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if saveto is not None:
        plt.savefig(saveto)
        plt.close()
    return

def plot_multiwell_trajectories(
        xycoord, well_ids, worm_ids, subsampling_rate=1, title=None,
        wells_to_plot=None, bad_wells=None, well_size=None, saveto=None,
        well_title=None, n_wells=96):
    """
    Plot all the worm trajectories of a multiwell plate.

    Parameters
    ----------
    xycoord : dataframe shape = ( sum(trajectories_lengths) , 2)
        A dataframe with 'coord_x' and 'coord_y' columns containing all the
        trajectories timeseries of all the wells of the multi-well plate.
    well_ids : array-like shape = (sum(trajectories_lengths), )
        Contains the well id for each row of xycoord.
    worm_ids : array-like shape = (sum(trajectories_lengths), )
        Contains the worm if for each row of xycoord.
    subsampling_rate : int, optional
        Subsampling rate for plotting purposes. The default is 1.
    title : string, optional
        The title of the plot. The default is None.
    wells_to_plot : list of well ids, optional
        It specifies the wells to plot. The default is None.
    bad_wells : list of well ids, optional
        It specified wells that will be marked as bad in the plot.
        The default is None.
    well_size : float, optional
        The diameter of each well (in the same units as the x,y coordinates).
        The default is None.
    saveto : path, optional
        The full path to the file where the figure will be saved. If None, the
        figure will not be saved and will be kept open. The default is None.
    well_title : dictionary, optional
        A dictionary mathcing well ids to well titles. It defins the title of
        each individual well (subplot) in the figure. If None, the title of
        each subplot is the number of trajectories plotted.
        The default is None.

    Returns
    -------
    None.

    """
    import numpy as np
    import textwrap

    if n_wells==96:
        pos = pos96wellplate
    else:
        raise ValueError('Only 96 well plates supported.')

    xycoord = pd.DataFrame(xycoord)

    if wells_to_plot is None:
        wells_to_plot = np.sort(well_ids.unique())
        wells_to_plot = [w for w in wells_to_plot if 'n/a' not in w]
    if bad_wells is None:
        bad_wells = []

    fig, ax = plt.subplots(4, 4, figsize=(10,10))
    plt.subplots_adjust(hspace = 0.6, wspace=0.6)
    for axes in ax.flatten():
        axes.set_xticks([])
        axes.set_yticks([])
    for iw, well in enumerate(np.sort(well_ids.unique())):
        if 'n/a' in well:
            continue

        i = pos[well]
        wdf = xycoord.loc[well_ids==well, :]
        wwormids = worm_ids[well_ids==well]
        n_traj = np.unique(wwormids).shape[0]

        xlim = [wdf['coord_x'].min(), wdf['coord_x'].max()]
        ylim = [wdf['coord_y'].min(), wdf['coord_y'].max()]
        if well_size is None:
            xlim = [xlim[0]*0.95, xlim[1]*1.05]
            ylim = [ylim[0]*0.95, ylim[1]*1.05]
        else:
            margin = (well_size - xlim[1] + xlim[0]) / 2
            xlim = [xlim[0]-margin, xlim[1]+margin]
            margin = (well_size - ylim[1] + ylim[0]) / 2
            ylim = [ylim[0]-margin, ylim[1]+margin]

        if well in wells_to_plot:
            if well_title is None:
                title = 'well {} : {} trajectories'.format(well, n_traj)
            else:
                title = textwrap.fill(well_title[well], 25)

            ax[i[0],i[1]].set_title(title, fontsize=8)
            wdf.groupby(by=wwormids).apply(
                _plot_trajectory, axes=ax[i[0],i[1]], subsampling_rate=subsampling_rate
                )
        elif well in bad_wells:
            if well_title is None:
                title = 'BAD well {} : {} trajectories'.format(well, n_traj)
            else:
                title = textwrap.fill(well_title[well], 25)

            ax[i[0],i[1]].set_title(title, fontsize=8)
            wdf.groupby(by=wwormids).apply(
                _plot_trajectory, axes=ax[i[0],i[1]],
                subsampling_rate=subsampling_rate, color='grey'
                )
            ax[i[0],i[1]].plot(xlim, ylim, ':', c='black')
            ax[i[0],i[1]].plot(xlim, np.flip(ylim), ':', c='black')

        ax[i[0],i[1]].set_xlim(xlim)
        ax[i[0],i[1]].set_ylim(ylim)
        if well_size is None:
            ax[i[0],i[1]].axis('equal')


    if saveto is not None:
        plt.savefig(saveto)
        plt.close()

    return

#%% helper functions
def _plot_trajectory(
        xycoord, axes=None, subsampling_rate=1, color=None,
        xlim=None, ylim=None):

    xycoord = xycoord.iloc[::subsampling_rate, :]

    if axes is None:
        fig, axes = plt.subplots()

    axes.plot(*xycoord.values.T, color=color)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    return


if __name__=="__main__":

    #"/Volumes/behavgenom$/Ida/Data/Hydra/SyngentaScreen/Results/" + \
    file = "/Users/em812/Data/Drugs/StrainScreens/SyngentaScreen/Results/" + \
        "20191206/syngenta_screen_run1_bluelight_20191206_144342.22956809/" + \
        "metadata_featuresN.hdf5"

    well_name = pd.read_hdf(file, key='timeseries_data', mode='r')['well_name']

    df = pd.read_hdf(file, key='trajectories_data', mode='r')

    df.insert(0, 'well_name', well_name)

    print(df.shape)
    df = df[df['was_skeletonized'].astype(bool)]
    print(df.shape)
    df = df[df['well_name']!='n/a']
    print(df.shape)

    # for wormid in df['worm_index_joined'].unique()[:19]:
    #     plot_trajectory(
    #         df.loc[df['worm_index_joined']==wormid, ['coord_x', 'coord_y']],
    #         subsampling_rate=25,
    #         xlim=[df['coord_x'].min()*1.05, df['coord_x'].max()*1.05],
    #         ylim=[df['coord_y'].min()*1.05, df['coord_y'].max()*1.05])

    # for well in well_name.unique():
    #     n_traj = df.loc[df['well_name']==well, 'worm_index_joined'].nunique()
    #     plot_well_trajectories(
    #         df.loc[df['well_name']==well, ['coord_x', 'coord_y']],
    #         df.loc[df['well_name']==well, 'worm_index_joined'],
    #         subsampling_rate=25, title='well {} : {} trajectories'.format(well, n_traj),
    #         xlim=[df.loc[df['well_name']==well, 'coord_x'].min()*0.95,
    #               df.loc[df['well_name']==well, 'coord_x'].max()*1.05],
    #         ylim=[df.loc[df['well_name']==well, 'coord_y'].min()*0.95,
    #               df.loc[df['well_name']==well, 'coord_y'].max()*1.05]
    #         )

    plot_multiwell_trajectories(
        df[['coord_x', 'coord_y']], df['well_name'], df['worm_index_joined'],
        wells_to_plot = ['H11', 'H9', 'H10', 'G10', 'G9', 'G11'],
        subsampling_rate=25
        )