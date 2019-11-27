#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:44:10 2019
@author: lferiani
"""

import pandas as pd
import imgstore
import json
from pathlib import Path
from matplotlib import pyplot as plt

# %% class definition


class ExtraDataReader(object):

    def __init__(self, filename):
        self.store = imgstore.new_for_filename(filename)
        self.ext = '.extra_data.json'
        self.extra_data = None

    def _get_extra_data(self):
        """Only called by the __init__"""
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

        return out

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


# %%

if __name__ == '__main__':
    fname = Path('/Volumes/behavgenom$/Luigi/')
    fname = fname / 'Data/Blue_LEDs_tests/RawVideos/20191017_pilotexp/'
    fname = fname / 'pilotexp_run1_bluelight_20191017_155329.22956813'

    edr = ExtraDataReader(fname)
    print(edr.get_extra_data(includeonly=['light']))
    edr.plot_extra_data()
    sumedr = edr.get_summary(includeonly=['light'])
    print(sumedr)
