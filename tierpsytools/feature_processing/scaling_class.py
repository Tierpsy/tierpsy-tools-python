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
        self.std_ = None
        self.mean_ = None
        self.min_ = None
        self.diff_ = None
        if axis is None and scaling == 'normalize':
            self.axis_ = 1
        else:
            self.axis_ = 0
        self.norm_ = norm
        self._fitted = False

    ## Define class methods

    # normalization
    def fit(self,Xin):

        # Check if already fitted
        if self._fitted:
            print('Warning: The scaling class instance has already been fitted. The results will be overwritten.')

        if self.scaling is None:
            return

        # Check input
        if isinstance(Xin,list):
            X = np.array(Xin[:])
        elif isinstance(Xin,pd.DataFrame):
            X = Xin.values
        elif isinstance(Xin,np.ndarray):
            X = np.copy(Xin)
            pass
        else:
            ValueError('Data type not recognised in scalingClass. Input can be list, numpy array or pandas dataframe.')

        # Fit
        if self.scaling == 'standardize':
            self.mean_ = np.mean(X,axis=self.axis_)
            self.std_ = np.std(X,axis=self.axis_)
        elif self.scaling == 'minmax_scale':
            self.min_ = np.min(X,axis=self.axis_)
            self.diff_ = np.max(X,axis=self.axis_)-np.min(X,axis=self.axis_)
        elif self.scaling == 'normalize':
            pass
        else:
            ValueError('Scaling type not recognised by scalingClass.')

        self._fitted = True

    def fit_transform(self, Xin):

        if self.scaling is None:
            return Xin

        # Check if already fitted
        if self._fitted:
            print('Warning: The scaling class instance has already been fitted. The results will be overwritten.')

        # Check input
        isdataframe=False
        if isinstance(Xin,list):
            X = np.array(Xin[:])
        elif isinstance(Xin,pd.DataFrame):
            columns = Xin.columns
            index = Xin.index
            isdataframe=True
            X = Xin.copy().values
        elif isinstance(Xin,np.ndarray):
            X=np.copy(Xin)
        else:
            ValueError('Data type not recognised in scalingClass. Input can be list, numpy array or pandas dataframe.')

        if self.axis_==1:
            X = X.T

        if self.scaling == 'standardize':
            self.mean_ = np.mean(X,axis=0)
            self.std_ = np.std(X,axis=0)
            X[:,self.std_!=0] = (X[:,self.std_!=0]-self.mean_[self.std_!=0])/self.std_[self.std_!=0]
            X[:,self.std_==0] = 0.0

        elif self.scaling == 'minmax_scale':
            self.min_ = np.min(X,axis=0)
            self.diff_ = np.max(X,axis=0)-np.min(X,axis=0)
            X[:,self.diff_!=0] = (X[:,self.diff_!=0]-self.min_[self.diff_!=0])/self.diff_[self.diff_!=0]
            X[:,self.diff_==0] = 0.5

        elif self.scaling == 'normalize':
            if self.norm_ == 'l1':
                norms = np.abs(X).sum(axis=0)
            elif self.norm_ == 'l2':
                norms = np.linalg.norm(X, axis=0)
            elif self.norm_ == 'max':
                norms = np.max(X, axis=0)
            norms = _handle_zeros_in_scale(norms, copy=False)
            X = np.divide(X,norms)
            self.norms_ = norms

        else:
            ValueError('Scaling type not recognised by scalingClass.')

        if self.axis_==1:
            X=X.T

        if isdataframe:
            X = pd.DataFrame(X,columns=columns,index=index)

        self._fitted = True

        return X

    def transform(self, Xin):

        if self.scaling is None:
            return Xin

        # Check if already fitted
        if not self._fitted:
            ValueError('The scaling class instance has not been fitted. Use the fit method befor using the transofrm method.')

        isdataframe=False
        if isinstance(Xin,list):
            X = np.array(Xin[:])
        elif isinstance(Xin,pd.DataFrame):
            columns = Xin.columns
            index = Xin.index
            isdataframe=True
            X = Xin.copy().values
        elif isinstance(Xin,np.ndarray):
            X = np.copy(Xin)
        else:
            ValueError('Data type not recognised in scalingClass. Input can be list, numpy array or pandas dataframe.')

        if self.axis_==1:
            X = X.T

        if self.scaling == 'standardize':
            if self.mean_ is not None and self.std_ is not None:
                X[:,self.std_!=0] = (X[:,self.std_!=0]-self.mean_[self.std_!=0])/self.std_[self.std_!=0]
                X[:,self.std_==0] = 0.0
            else:
                ValueError('The instance of the scalingClass must be fitted before being used to transform a feature matrix. Use the fit class method first.')
        elif self.scaling == 'minmax_scale':
            if self.min_ is not None and self.diff_ is not None:
                X[:,self.diff_!=0] = (X[:,self.diff_!=0]-self.min_[self.diff_!=0])/self.diff_[self.diff_!=0]
                X[:,self.diff_==0] = 0.5
            else:
                ValueError('The instance of the scalingClass must be fitted before being used to transform a feature matrix. Use the fit class method first.')
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
        else:
            ValueError('Scaling type not recognised by scalingClass.')

        if self.axis_==1:
            X = X.T

        if isdataframe:
            X = pd.DataFrame(X,columns=columns,index=index)
        return X
