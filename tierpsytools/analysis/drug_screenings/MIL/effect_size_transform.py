#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:51:28 2020

@author: em812
"""
from warnings import warn
import numpy as np
import pandas as pd
from tierpsytools.feature_processing.scaling_class import scalingClass
from tierpsytools.analysis.drug_screenings.drug_class import drugClass_v1
from time import time
import pdb

def bags_to_effect_size_params(Xs, metas, inparams, mapper):
    """
    Takes bags of tierpsy features (list of tierpsy features sets for
    the same drugs) and creates one dataset of effect size parameters.
    If the number of bags given is n, then each drug will have n rows of
    effect size parameters in the returned dataset P.
    """
    P = []; y = []; drug = []
    for X, xmeta in zip(Xs, metas):
        transformer = DrugEffectSizeParams(**inparams)
        ip, iy, idr = transformer.fit_transform(
            X, xmeta['drug_type'], xmeta['drug_dose'],
            xmeta['drug_type'].map(mapper).values, control='DMSO',
            return_scaled=False)
        P.append(ip)
        y.append(iy)
        drug.append(idr)

    P = np.concatenate(P, axis=0)
    y = np.concatenate(y)
    drug = np.concatenate(drug)

    return P, y, drug

class effectSizeTransform():

    """
    Transforms tierpsy features for multiple drugs at multiple doses
    (generally multiple replicates per dose) to a set of effect size
    parameters. Each drug is characterised by a single vector
    of effect size parameters.
    """

    valid_effect_types = ['max', 'mean', 'median']

    ## Initiate class
    def __init__(
            self,
            effect_type = ['max','mean'],
            cutoff = None,
            normalize_effect = None,
            binary_effect = False,
            scale_per_compound = False,
            scale_samples = None, #'minmax_scale', #'scale' #normalize #
            scale_params = None
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
        self.parameters_mask = None
        self.param_scaler = self.scaler(scale_params)


    def scaler(self, scalingtype):

        if isinstance(scalingtype, str) or scalingtype is None:
            scalerinstance = scalingClass(scaling = scalingtype)
        elif hasattr(scalingtype, 'fit'):
            scalerinstance = scalingtype
        else:
            raise ValueError(
                'Scaling parameter type not recognised. Valid parameter types '+
                'include strings \'minmax_scale\', \'standardize\' and \'normalize\' '+
                'and instances of scaling classes with a fit() method.')

        return scalerinstance


    def scale_individual_compound(self, feat, control):

        # Tranform only if the scaler is doing something
        if self.scale_per_compound:
            sample_scaler = self.scaler(self.scale_samples)

            features = sample_scaler.fit_transform(
                pd.concat([control, feat], axis=0, sort=False).values)

            # Separate dmso from drug doses
            control = pd.DataFrame(
                features[:control.shape[0], :],
                columns = control.columns,
                index = control.index)

            feat = pd.DataFrame(
                features[control.shape[0]:, :],
                columns = feat.columns,
                index = feat.index)

        return feat, control

    def _transform(self, feat, dose, control):
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
            normalize_effect = ['control_mean', 'control_std', None] how to normalize
                the effect of each feature. If None, the effect if not
                normalized.
            normalize_features = function object for normalization of feature
                matrix
        return:
            parameters = effect parameters
        """

        # make sure the fetures match between control and drug
        assert control.shape[1] == feat.shape[1]
        assert all([ft in control.columns for ft in feat.columns])
        assert all([ft in feat.columns for ft in control.columns])

        # Scale the features
        feat, control = self.scale_individual_compound(feat, control)

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

        return parameters

    def check_input(self, bags, control, doses):
        from tierpsytools.analysis.drug_screenings.drug_class import drugClass

        if (isinstance(bags[0], np.ndarray) and not isinstance(control, np.ndarray)) \
            or (isinstance(bags[0], pd.DataFrame) and not isinstance(control, pd.DataFrame)) \
            or (isinstance(bags[0], drugClass) and not isinstance(control, pd.DataFrame)):
                raise ValueError('Bags and control are not the same type.')

        assert all([bags[i].feat.shape[1]==bags[i-1].feat.shape[1]
                    for i in range(len(bags))]), \
            'The number of features is not constant among bags.'

        assert bags[0].feat.shape[1] == control.shape[1], \
            'The control has different number of features than the bags.'

        if not isinstance(bags[0], drugClass):
            if isinstance(bags[0], pd.DataFrame):
                if isinstance(doses, str) and doses!='index':
                    bags = [bag.set_index(doses) for bag in bags]
            elif isinstance(bags[0], np.ndarray):
                if doses is None or isinstance(doses, str):
                    raise ValueError(
                        'The parameter doses need to be defined with the '
                        'drug dose for each sample of every bag.')
                else:
                    bags = [pd.DataFrame(bag, index=dose)
                            for bag,dose in zip(bags, doses)]

        if isinstance(control, np.ndarray):
            control = pd.DataFrame(control)

        return bags, control

    def fit(self, bags, control, doses=None, update_druginstances=False):
        """
        Parameters:
            doses : None, str or list-like
                    If bags is a list of drugClass instances, then this
                    is ignored.
                    If bags is a list of pd.DataFrames, then doses should be a
                    string, either 'index' (if the dose information is in the
                    index of the dataframe) or the name of the column that
                    contains the dose information.
                    If bags is a list of numpy arrays, then doses should be
                    a list of arrays with the same number of elements as the
                    number of samples in each bag., containing the dose
                    information for each sample.

        """
        from tierpsytools.analysis.drug_screenings.drug_class import drugClass, drugClass_v1

        bags, control = self.check_input(bags, control, doses)

        parameters = np.zeros((len(bags), control.shape[1]*self.n_param))
        for ibag, bag in enumerate(bags):
            if isinstance(bag, (drugClass, drugClass_v1)):
                feat = bag.feat
                dose = bag.drug_dose
            else:
                feat = bag
                dose = bag.index.to_list()

            parameters[ibag, :] =  self._transform(feat, dose, control)

            if update_druginstances:
                bag.mil_transform = parameters[ibag, :]

        self.effect_size_parameters = parameters

        return

    def fit_transform(self, bags, control, doses=None, update_druginstances=False):

        self.fit(bags, control, doses=doses, update_druginstances=update_druginstances)

        return self.effect_size_parameters

    def scale_parameters(self, apply_mask=False):

        if apply_mask and self.parameters_mask is not None:
            parameters = self.effect_size_parameters[:, self.parameters_mask]
        else:
            parameters = self.effect_size_parameters

        return self.param_scaler.fit_transform(parameters)


    def classify(self,
            bags, control,
            bagLabels, estimator,
            doses=None, update_druginstances=False):

        if not 'effect_size_parameters' in self.__dict__:
            self.fit(
                bags, control, doses=doses,
                update_druginstances=update_druginstances)

        parameters = self.effect_size_parameters

        if self.parameters_mask is None:
            self.parameters_mask = np.std(self.effect_size_parameters, axis=0)!=0

        parameters = self.scale_parameters(apply_mask=True)

        self.classifier = estimator.fit(parameters, bagLabels)

        return


class DrugEffectSizeParams(effectSizeTransform):
    """
    Wrapper for the effectSizeTransform class that handles pandas dataframes
    and pandas series as input.
    """
    def __init__(
            self,
            effect_type = ['max','mean'],
            cutoff = None,
            normalize_effect = None,
            binary_effect = False,
            scale_per_compound = False,
            scale_samples = None, #'minmax_scale', #'scale' #normalize #
            scale_params = None):

        super().__init__(effect_type, cutoff, normalize_effect,
            binary_effect, scale_per_compound, scale_samples,
            scale_params)

    def fit(self, feat, drug, dose, moa, control='DMSO'):

        # Create drug instances
        namelist = np.unique(drug)
        namelist = namelist[namelist!=control]

        # get feature names
        cols = feat.columns.to_numpy()
        # get parameters names
        p_cols = ['_'.join([col, eff]) for col in cols for eff in self.effect_type]

        druglist = [drugClass_v1(
            feat[drug==idg], idg, dose[drug==idg], MOAinfo=moa[drug==idg])
            for idg in namelist]

        control_feat = feat[drug==control]

        est = effectSizeTransform(
            self.effect_type, self.cutoff, self.normalize_effect,
            self.binary_effect, self.scale_per_compound, self.scale_samples,
            self.param_scaler)
        est.fit(druglist, control_feat)

        if  est.effect_size_parameters.shape[1]==0:
            pdb.set_trace()
            raise Exception('empty parameters matrix')

        scaled_params = est.scale_parameters(apply_mask=False)

        self.P = pd.DataFrame(est.effect_size_parameters, columns=p_cols)
        self.P_scaled = pd.DataFrame(scaled_params, columns=p_cols)
        self.y = np.array([drug.moa['MOA_group'] for drug in druglist])
        self.drugs = namelist

        return

    def fit_transform(self, feat, drug, dose, moa, control='DMSO', return_scaled=True):
        if not hasattr(self, 'P'):
            self.fit(feat, drug, dose, moa, control=control)

        if return_scaled:
            return self.P_scaled, self.y, self.drugs
        else:
            return self.P, self.y, self.drugs

    def get_parameters(self, scaled=False):
        if scaled:
            return self.P_scaled
        else:
            return self.P


