#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:35:57 2019

@author: em812
"""

import numpy as np
import pandas as pd


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale

class scalingClass():
    """
    A class for scaling features sets. It acceptes the following options:
        - standardize : ( X - X.mean() ) / X.std()
        - normalize : X = X / norm( X )
          The norm can be 'l1', 'l2' or 'max'
        - minmax_scale : ( X - X.min() ) / ( X.max() - X.min() )
    """

    ## Initiate class
    def __init__(self, scaling='standardize', axis=None, norm='l2'):
        ## Define class attributes
        self.scaling = scaling

        if axis is None:
            if scaling == 'normalize':
                self.axis_ = 1
            else:
                self.axis_ = 0
        else:
            self.axis_ = axis
        self.norm_ = norm
        self._fitted = False

    ## Define class methods

    def check_input(self, Xin):
        isdataframe=False
        if isinstance(Xin,list):
            X = np.array(Xin[:])
        elif isinstance(Xin,pd.DataFrame):
            isdataframe=True
            X = Xin.copy().values
        elif isinstance(Xin,np.ndarray):
            X = np.copy(Xin)
        else:
            ValueError('Data type not recognised in scalingClass. Input can be list, numpy array or pandas dataframe.')
        return X, isdataframe

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        if self._fitted:
            if hasattr(self, 'mean_'):
                del self.mean_
                del self.std_
            if hasattr(self, 'min_'):
                del self.min_
                del self.diff_
            if hasattr(self, 'norm_'):
                del self.norm_

    # normalization
    def fit(self, Xin, y=None):

        # Check if already fitted
        if self._fitted:
            print('Warning: The scaling class instance has already been '+
                  'fitted. The results will be overwritten.')
            # Reset internal state before fitting
            self._reset()

        if self.scaling is None:
            return

        # Check input
        X, _ = self.check_input(Xin)


        # Fit
        if self.scaling == 'standardize' or self.scaling == 'rescale1' or self.scaling == 'rescale2':
            self.mean_ = np.mean(X,axis=self.axis_)
            self.std_ = np.std(X,axis=self.axis_)
        elif self.scaling == 'minmax_scale':
            self.min_ = np.min(X,axis=self.axis_)
            self.diff_ = np.max(X,axis=self.axis_)-np.min(X,axis=self.axis_)
        elif self.scaling == 'normalize':
            if self.axis_==0:
                if self.norm_ == 'l1':
                    norms = np.abs(X).sum(axis=0)
                elif self.norm_ == 'l2':
                    norms = np.linalg.norm(X, axis=0)
                elif self.norm_ == 'max':
                    norms = np.max(X, axis=0)
                norms = _handle_zeros_in_scale(norms, copy=False)
                self.norms_ = norms
            else:
                self.norms_ = None
        else:
            ValueError('Scaling type not recognised by scalingClass.')

        self._fitted = True

    def transform(self, Xin, y=None):

        if self.scaling is None:
            return Xin

        # Check if already fitted
        if not self._fitted:
            ValueError(
                'The scaling class instance has not been fitted. Use the '+
                'fit method before using the transofrm method.')

        # Check input
        X, isdataframe = self.check_input(Xin)

        if self.axis_==1:
            X = X.T

        if self.scaling == 'standardize':
            X[:,self.std_!=0] = (X[:,self.std_!=0]-self.mean_[self.std_!=0])/self.std_[self.std_!=0]
            X[:,self.std_==0] = 0.0

        elif self.scaling == 'minmax_scale':
            X[:,self.diff_!=0] = (X[:,self.diff_!=0]-self.min_[self.diff_!=0])/self.diff_[self.diff_!=0]
            X[:,self.diff_==0] = 0.5

        elif self.scaling == 'normalize':
            if self.axis_ == 0:
                X = np.divide(X,self.norms_)
            else:
                if self.norm_ == 'l1':
                    norms = np.abs(X).sum(axis=0)
                elif self.norm_ == 'l2':
                    norms = np.linalg.norm(X, axis=0)
                elif self.norm_ == 'max':
                    norms = np.max(X, axis=0)
                norms = _handle_zeros_in_scale(norms, copy=False)
                X = np.divide(X,norms)

        elif self.scaling == 'rescale1':
            X[:,self.std_!=0] = (X[:,self.std_!=0]-self.mean_[self.std_!=0])/self.std_[self.std_!=0]
            X[:,self.std_==0] = 0.0

            norms = np.linalg.norm(X, axis=1)
            norms = _handle_zeros_in_scale(norms, copy=False)
            X = np.divide(X.T, norms).T

        elif self.scaling == 'rescale2':
            X[:,self.std_!=0] = (X[:,self.std_!=0]-self.mean_[self.std_!=0])/self.std_[self.std_!=0]
            X[:,self.std_==0] = 0.0

            X = X.T
            min_ = np.min(X, axis=0)
            diff_ = np.max(X, axis=0) - np.min(X, axis=0)
            X[:, diff_!=0] = (X[:, diff_!=0] - min_[diff_!=0])/diff_[diff_!=0]
            X[:, diff_==0] = 0.5
            X = X.T

        else:
            ValueError('Scaling type not recognised by scalingClass.')

        if self.axis_==1:
            X = X.T

        if isdataframe:
            X = pd.DataFrame(X, columns=Xin.columns, index=Xin.index)
        return X


    def fit_transform(self, Xin, y=None):

        if self.scaling is None:
            return Xin

        # Check if already fitted
        self.fit(Xin)
        self._fitted = True

        return self.transform(Xin)


