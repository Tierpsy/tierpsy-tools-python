#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:44:10 2019
@author: lferiani
"""

import json
import tqdm
import imgstore
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tierpsytools.hydra.hydra_filenames_helper import find_imgstore_videos

warnings.filterwarnings('ignore', message='calling yaml.load()*')


# %% add functionality to Path
def strreplace(self, old, new):
    return Path(str(self).replace(old, new))


Path.strreplace = strreplace


# %% class definition
class ExtraDataReader(object):
    """"ExtraDataReader

    Wrapper for loopbio's own class as I was having problems with it'
    """

    def __init__(self, filename):
        try:
            self.store = imgstore.new_for_filename(str(filename))
        except:
            self.store = None
        self.filename = filename
        print(self.filename)
        self.ext = '.extra_data.json'
        self.extra_data = None

    def _get_extra_data(self):
        """Only called by the __init__"""
        if self.store is None:
            extra_data_fnames = list(self.filename.parent.glob('*' + self.ext))
            print(extra_data_fnames)
        else:
            extra_data_fnames = [chunk+self.ext
                                 for (_, chunk)
                                 in self.store._chunk_n_and_chunk_paths]
        dfs = []
        for extra_data in extra_data_fnames:
            with open(extra_data) as fid:
                dct = json.load(fid)
            dfs.append(pd.DataFrame.from_dict(dct))
        df = pd.concat(dfs, axis=0, ignore_index=True)
        t0 = df['recording_start'].unique()[0]
        df.insert(0, 'time', df['frame_time'] - t0)
        return df

    def get_extra_data(self, includeonly=[]):
        """
        Public method, returns the stored extra_data and,
        if for some reason that is empty, reads the extra data all over again
        """
        if self.extra_data is None:
            self.extra_data = self._get_extra_data()
        assert isinstance(includeonly, list), '"includeonly" has to be a list'
        if len(includeonly) > 0:
            if 'time' not in includeonly:
                includeonly.insert(0, 'time')
            out = self.extra_data[includeonly]
        else:
            out = self.extra_data

        return out.copy()

    def plot_extra_data(self):

        fig, axs = plt.subplots(nrows=2,
                                ncols=2,
                                gridspec_kw={'wspace': 0,
                                             'hspace': 0,
                                             'left': 0.1,
                                             'right': 0.9,
                                             'top': 0.93})
        plt.suptitle(self.store.filename.split('/')[-1])
        self.extra_data['tempi'].plot(x='time',
                                      ax=axs[0, 0],
                                      ylim=(15, 35),
                                      label='in')
        self.extra_data['tempo'].plot(x='time',
                                      ax=axs[0, 0],
                                      ylim=(15, 35),
                                      label='out')
        axs[0, 0].tick_params(labelbottom=False)
        axs[0, 0].set_ylabel(u'Temperature, [\u00B0C]')
        axs[0, 0].legend(frameon=False)
        self.extra_data['humidity'].plot(x='time',
                                         ax=axs[0, 1],
                                         ylim=(20, 90))
        axs[0, 1].tick_params(labelbottom=False)
        axs[0, 1].yaxis.tick_right()
        axs[0, 1].yaxis.set_label_position('right')
        axs[0, 1].set_ylabel(u'Humidity, [%]')
        self.extra_data['light'].plot(x='time',
                                      ax=axs[1, 0])
        axs[1, 0].set_ylabel(u'light intensity, [a.u.]')
        axs[1, 0].set_xlabel(u'Time, [s]')
        self.extra_data['vin'].plot(x='time',
                                    ax=axs[1, 1])
        axs[1, 1].yaxis.tick_right()
        axs[1, 1].yaxis.set_label_position('right')
        axs[1, 1].set_ylabel(u'vin, [a.u.]')
        axs[1, 1].set_xlabel(u'Time, [s]')

        for ax in axs.flatten():
            ax.tick_params(direction='in')

        return

    def get_summary(self, includeonly=[]):
        assert isinstance(includeonly, list), '"includeonly" has to be a list'
        quantities = ['tempi', 'tempo', 'light', 'humidity', 'vin']
        if len(includeonly) > 0:
            quantities = [q for q in quantities if q in includeonly]
        out = self.get_extra_data()[quantities].describe()
        return out.loc[['mean', 'std', 'min', 'max']]


def plot_imagingset_sensordata(setname, setdata):
    # first read and collate all data
    edr_df = []  # initialising list of dataframes
    frame = None
    for i, row in setdata.iterrows():
        # use the class above
        edr = ExtraDataReader(row['full_path'])
        df = edr.get_extra_data(includeonly=['light', 'tempi',
                                             'tempo', 'humidity'])
        df['channel'] = row['channel']
        edr_df.append(df)
        # read first frame as well
        if row['channel'] == 'Ch1':
            try:
                frame = edr.store.get_next_image()[0]
                edr.store.close()
                is_first_frame = True
            except:
                print('No file found')
                is_first_frame = False
    # make one big df
    edr_df = pd.concat(edr_df, axis=0, ignore_index=True)

    # Preprocessing:
    # fix light sensor issues
    edr_df['light_was_nan'] = edr_df['light'].isna()
    # simple close one-point holes in the sensor data
    edr_df['light'].fillna(method='ffill', limit=1, inplace=True)
    # assume that if failed for more than 1 s then it's the light's fault
    sat_value = 8000
    edr_df['light'].fillna(sat_value, inplace=True)
    edr_df['light'].clip(lower=0, upper=sat_value, inplace=True)
    # round time to the second
    edr_df['time_s'] = edr_df['time'].round()

    # Plots
    fig, axs = plt.subplots(2, 2, figsize=(8, 4.8))
    # light
    sns.lineplot(x='time_s',
                 y='light',
                 data=edr_df,
                 estimator='mean',
                 ci='sd',
                 ax=axs[0, 0])
    sns.scatterplot(x='time', y='light',
                    data=edr_df[edr_df['light_was_nan']],
                    marker='x',
                    color="tab:orange",
                    edgecolor=None,
                    label="was_nan",
                    ax=axs[0, 0])
    axs[0, 0].set_ylim((-500, 8500))
    axs[0, 0].tick_params(labelbottom=False)
    # temperatures
    sns.lineplot(x='time_s',
                 y='temperature',
                 hue='sensor',
                 estimator='mean',
                 ci='sd',
                 ax=axs[0, 1],
                 data=edr_df.melt(id_vars=['time_s'],
                                  value_vars=['tempi', 'tempo'],
                                  value_name='temperature',
                                  var_name='sensor'))
    axs[0, 1].set_ylim((16, 28))
    # axs[0, 1].yaxis.tick_right()
    axs[0, 1].tick_params(labelleft=False, labelright=True)
    axs[0, 1].yaxis.set_label_position("right")
    # humidity
    sns.lineplot(x='time_s',
                 y='humidity',
                 estimator='mean',
                 ci='sd',
                 ax=axs[1, 0],
                 data=edr_df)
    axs[1, 0].set_ylim((15, 60))
    plt.setp(axs[1, 0].get_yticklabels()[-1], visible=False)
    # part of frame with name
    if frame is not None:
        axs[1, 1].imshow(np.rot90(frame[:, min(frame.shape):]), cmap='gray')
    axs[1, 1].set_axis_off()
    # title figure and otehr adjustments
    plt.subplots_adjust(hspace=.0)
    for ax in axs.flatten():
        ax.tick_params(direction='in', top=True, right=True)
    fig.suptitle(setname)

    return fig


def check_hydra_sensordata(path_to_imgstores, is_makereport=True):
    """check_hydra_sensordata
    Read the sensor data collected by the Hydra rigs and returns a report
    showing how light, temperature, humidity behaved during the recording.

    Parameters
    ----------
    path_to_imgstores : pathlib.Path or string
        Path pointing to the directory containing the recordings.

    Returns
    -------
    None.

    """
    # input check
    if isinstance(path_to_imgstores, str):
        path_to_imgstores = Path(path_to_imgstores)
    # create the report's path
    # check for standard folder structure
    if 'RawVideos' in path_to_imgstores.parts:
        path_to_report = path_to_imgstores.strreplace('RawVideos',
                                                      'AuxiliaryFiles')
        path_to_report.mkdir(parents=True, exist_ok=True)
    else:
        path_to_report = path_to_imgstores
    path_to_report = path_to_report / 'sensors_data.pdf'

    # get a dataframe with all videos
    videos_df = find_imgstore_videos(path_to_imgstores)

    # group all recordings by imaging name (to get the 6 cameras' videos from
    # one recording), loop over them, and collect the data. Store the data in
    # a list of dataframes that will then colated in a single dataframe
    videos_df_g = videos_df.groupby("imaging_set")
    # loop on imaging sets
    with PdfPages(path_to_report, keep_empty=False) as pdf:
        for name, group in tqdm.tqdm(videos_df_g):
            # function that acts on the imaging set
            fig = plot_imagingset_sensordata(name, group)
            if is_makereport:
                pdf.savefig(fig)
                plt.close(fig)
    if not is_makereport:
        plt.show(block=False)
        input("Press any key to continue. This will close all figures.")
        plt.close('all')


def hydra_sensordata_report():
    # input parser
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path',
                        type=str)
    parser.add_argument('-n', '--no-pdf',
                        action='store_true')  # no_pdf defaults to false
    args = parser.parse_args()
    dirname = Path(args.folder_path)
    is_makereport = not args.no_pdf  # dash is converted to underscore
    # However dash is standard in *nix extra inputs to I'll keep it a dash
    check_hydra_sensordata(dirname, is_makereport=is_makereport)


# %% main


if __name__ == '__main__':
    # dirname = Path('/Volumes/behavgenom$/Ida/')
    # dirname = dirname / 'Data/Hydra/SygentaTestVideos/RawVideos/'

    # plt.close('all')
    # check_hydra_sensordata(dirname, is_makereport=True)
    hydra_sensordata_report()
    # edr = ExtraDataReader(fname)
    # print(edr.get_extra_data(includeonly=['light']))
    # edr.plot_extra_data()
    # sumedr = edr.get_summary(includeonly=['light'])
    # print(sumedr)
