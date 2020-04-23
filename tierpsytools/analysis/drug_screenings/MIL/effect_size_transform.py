#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:51:28 2020

@author: em812
"""

class effectSizeTransform():
    import numpy as np
    import pandas as pd

    valid_effect_types = ['max', 'mean', 'median']

    ## Initiate class
    def __init__(
            self,
            effect_type = ['max','mean'],
            cutoff = None,
            normalize_effect = None,
            binary_effect = False,
            scale_per_compound = False,
            scale_samples = 'minmax_scale', #'scale' #normalize #
            ):

        if isinstance(effect_type, str):
            effect_type = [effect_type]

        if any([tp not in self.valid_effect_types for tp in effect_type]):
            ValueError(
                'The effect_type can take the values \'max\',\'mean\' or '
                '\'median\'. The input value was not recognised.')

        if binary_effect and (cutoff is  None):
            raise ValueError(
                'Define a cutoff to obtain a binary effect transformation.')

        if binary_effect and normalize_effect is not None:
            warn('Binary effect parameters cannot be normalized. ' +
                 'The normalize_effect parameter will be ignored.')
            normalize_effect = None

        self.effect_type = effect_type
        self.binary_effect = binary_effect
        self.n_param = len(effect_type)
        self.cutoff = cutoff
        self.normalize_effect = normalize_effect
        self.scale_per_compound = scale_per_compound
        if scale_per_compound:
            self.scale_samples = scale_samples
        else:
            self.scale_samples = None


    def scale_individual_compound(self, feat):

        from inspect import isclass, isfunction

        # Scale compound
        if self.scale_per_compound:
            if isclass(self.scale_samples):
                scaler = self.scale_samples()
                features = scaler.fit_transform(feat)
            else:
                if isinstance(self.scale_samples, str):
                    if self.scale_samples == 'standardize':
                        from sklearn.preprocessing import scale
                        scaler = scale
                    elif self.scale_samples == 'minmax_scale':
                        from sklearn.preprocessing import minmax_scale
                        scaler = minmax_scale
                    elif self.scale_samples == 'normalize':
                        from sklearn.preprocessing import normalize
                        scaler = normalize
                elif isfunction(self.scale_samples):
                    scaler = self.scale_samples

                features = scaler(feat)

            feat = pd.DataFrame(
                features, columns=feat.columns, index=feat.index)

        return feat

    def transform(self, drugobject, control, update_drugobject=False):
        """
        Get parameters for the effect of an individual compound to all the
        features
        param:
            self = drug class containing the doses and the features for each
                dose
            effect_type = ['max','mean','median'] the way the effect if
                estimated based on all the data points at all doses.
                If the effect_type is an array like object with n options then
                n parameters are extracted.
            normalize_effect = ['dmso_mean','dmso_std',None] how to normalize
                the effect of each feature. If None, the effect if not
                normalized.
            normalize_features = function object for normalization of feature
                matrix
        return:
            parameters = effect parameters
        """
        # make sure the fetures match between control and drug
        assert control.shape[1] == drugobject.feat.shape[1]
        assert all([ft in control.columns for ft in drugobject.feat.columns])
        assert all([ft in drugobject.feat.columns for ft in control.columns])

        if self.scale_per_compound:
            feat = self.scale_individual_compound(
                pd.concat([control, drugobject.feat], axis=0, sort=False))

            # Separate dmso from drug doses
            control = feat.iloc[:control.shape[0], :]
            feat = feat.iloc[control.shape[0]:, :]
        else:
            feat = drugobject.feat[control.columns]

        dose = drugobject.drug_dose

        # Get effect size parameters
        n_param = self.n_param
        effect_type = self.effect_type
        n_feat = feat.shape[1]
        cutoff = self.cutoff

        parameters = np.empty((n_param,n_feat),dtype=float)
        for imeas,measure in enumerate(effect_type):
            if measure == 'max':
                diff = feat.groupby(by=dose).mean().values - \
                    control.mean().values
                ind_max_diff = np.argmax(np.abs(diff),axis=0)
                parameters[imeas] = diff[ind_max_diff, np.arange(n_feat)]
            elif measure == 'mean':
                parameters[imeas] = (feat.mean() - control.mean()).values
            elif measure == 'median':
                parameters[imeas] = (feat.median() - control.median()).values

            # Apply cutoff
            if self.binary_effect:
                bins = [
                    parameters[imeas] > cutoff * control.std(axis=0).values,
                    parameters[imeas] < -cutoff * control.std(axis=0).values
                    ]
                parameters[imeas] = np.select(bins, [1, -1], default=0)

            elif (not self.binary_effect) and (cutoff is not None):
                parameters[imeas] = np.select(
                    [np.abs(parameters[imeas]) < \
                     cutoff * control.std(axis=0).values],
                    [0.0], default=parameters[imeas])

            # Normalize parameters
            if self.normalize_effect is not None and not self.binary_effect:
                if self.normalize_effect == 'control_mean':
                    parameters[imeas] = \
                        parameters[imeas] / control.mean(axis=0).values
                elif self.normalize_effect == 'control_std':
                    parameters[imeas] = \
                        parameters[imeas] / control.std(axis=0).values

        parameters = parameters.reshape(1,-1,order='F').flatten()

        if update_drugobject:
            drugobject.mil_transform = parameters

        return parameters
