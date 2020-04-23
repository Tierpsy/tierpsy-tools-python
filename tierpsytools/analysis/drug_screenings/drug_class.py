# -*- coding: utf-8 -*-
"""
drug_class: define a drug class
Basic attributes:
    - drug name
    - drug doses
    - worm response feature matrix
    - feature normalization
Class variables:
    nb_of_drugs
    constant_features = features that are constant throughout the trajetory of any drug
Methods:
    - normalization (z-normalize, scale 0-1)
    - feature filtering
    - decomposition of feature matrix
Class method:
    - add features to the constant_features class variable

Created on Wed Aug 22 16:00:27 2018

@author: em812
"""

import numpy as np
from warning import warn

class drugClass():

    nb_of_drugs = 0
    constant_features = []

    ## Initiate class
    def __init__(
            self,
            feat, meta,
            MOAinfo=False,
            set_dose_as_index=False,
            drug_name_col = 'drug_name',
            drug_dose_col = 'drug_dose'
            ):

        ## Define class attributes
        names = meta[drug_name_col].unique()

        if names.shape[0]==0:
            self.drug_name = names[0]
        else:
            raise ValueError('More than one coumpound names found in metadata.')
        self.drug_dose = meta[drug_dose_col].astype(float)

        # If MOAinfo available, read data on compound class/group and mode of action
        if MOAinfo:
            # Read from metadata
            assert (meta[
                ['MOA_general', 'MOA_specific', 'MOA_group']
                ].nunique()==1).all()
            self.MOA_general, self.MOA_specific, self.MOA_general = meta[
                ['MOA_general', 'MOA_specific', 'MOA_group']].values[0]

        # keep feat as a dataframe and set row index = drug dose:
        self.feat = feat.copy()
        if set_dose_as_index:
            self.feat.set_index(meta['drug_dose'], inplace=True)
        self.normalization = None

        drugClass.nb_of_drugs += 1

    ## Define class methods

    # normalization
    def znormalize(self, updateClass=False):
        from sklearn.preprocessing import scale
        import pandas as pd

        if self.normalization is not None:
            warn('\nDrug feature matrix ' +
                'already normalized ({}). '.format(self.normalization) +
                'Call to znormalize() method ignored.')
            return self.feat
        else:
            mat = self.feat.copy()
            mat = scale(mat)
            mat = pd.DataFrame(mat, columns=self.feat.columns)
            if updateClass:
                self.feat = mat
                self.normalization = 'z-normalized'
        return mat

    def scale01(self, updateClass=False):
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd

        if self.normalization is not None:
            warn('\nDrug feature matrix ' +
                'already normalized ({}). '.format(self.normalization) +
                'Call to scale01() method ignored.')
            return self.feat
        else:
            mat = self.feat.copy()
            scaler = MinMaxScaler()
            mat = scaler.fit_transform(mat)
            mat = pd.DataFrame(mat,columns=self.feat.columns)
#            min_mat = np.min(mat,axis=0)
#            max_mat = np.max(mat,axis=0)
#            mat = mat-min_mat
#            mat = mat/(max_mat-min_mat)
            if updateClass:
                self.feat = mat
                self.normalization = 'scaled 0-1'
            return mat

    # feature filtering
    def feat_filter_const(self):
        """
        FEAT_FILTER_STD: remove features with zero std
        """
        # add the features to the constant feature list of the class (so they can be removed from all classes)
        cnst = self.feat.loc[:,self.feat.var()==0].columns.to_list()
        self.add_const_feat(cnst)
        # remove the features with std=0 from the feat matrix of the class instance
        self.feat = self.feat.loc[:,self.feat.var()!=0]

        return self

    def remove_constant_feat(self):
        """
        REMOVE_CONSTANT_FEAT: remove all the features in the class variable constant_features
        """
        self.feat = self.feat[self.feat.columns.difference(self.constant_features)]




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
        if self.binary_effect:
            return self._transform_binary(
                drugobject, control, update_drugobject=update_drugobject)

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
            if self.cutoff is not None:
                parameters[imeas][
                    np.abs(parameters[imeas]) < \
                        self.cutoff * control.std(axis=0).values
                    ] = 0.0
            # Normalize parameters
            if self.normalize_effect is not None:
                if self.normalize_effect == 'control_mean':
                    parameters[imeas] = \
                        parameters[imeas] / control.mean(axis=0).values
                elif self.normalize_effect == 'control_std':
                    parameters[imeas] = \
                        parameters[imeas] / control.std(axis=0).values

        parameters = parameters.reshape(1,-1,order='F').flatten()

        if update_drugobject:
            drugobject.effect_size_parameters = parameters

        return parameters


    def _transform_binary(
            self, effect_type='max', normalize_features=None, cutoff=0.5):
        """
        Get parameters for the effect of an individual compound to all the features
        param:
            self = drug class containing the doses and the features for each dose
            effect_type = ['max','mean','median'] the way the effect if estimated based on all the data points at all doses.
                            If the effect_type is an array like object with n options then n parameters are extracted.
            normalize_effect = ['dmso_mean','dmso_std',None] how to normalize the effect of each feature. If None, the effect if not normalized.
            normalize_features = function object for normalization of feature matrix
        return:
            parameters = effect parameters
        """
        import numpy as np
        import pandas as pd

        # Check input
        if isinstance(normalize_features,str):
            if normalize_features == 'scale':
                from sklearn.preprocessing import scale
                normalize_features = scale
            elif normalize_features == 'minmax_scale':
                from sklearn.preprocessing import minmax_scale
                normalize_features = minmax_scale
            elif normalize_features == 'normalize':
                from sklearn.preprocessing import normalize
                normalize_features = normalize
        if isinstance(effect_type,str):
            effect_type = [effect_type]
        elif not isinstance(effect_type,(np.ndarray,list)):
            ValueError('The effect_type must be a string [\'max\',\'mean\',\'median\'] or an array-like instance containing strings.')

        # Normalize features
        if normalize_features is not None:
            try:
                features = normalize_features.fit_transform(self.feat)
            except:
                features = normalize_features(self.feat)
            features = pd.DataFrame(features,columns=self.feat.columns,index=self.drug_dose.values)
        else:
            features = self.feat.set_index(self.drug_dose)

        # Separate dmso from drug doses
        dmso = features[self.drug_dose.values==0.0]
        if 'DMSO' not in self.drug_name:
            features = features[self.drug_dose.values!=0.0]

        # Get effect size parameters
        n_param = len(effect_type)
        n_feat = features.shape[1]
        parameters = np.empty((n_param,n_feat),dtype=float)
        for imeas,measure in enumerate(effect_type):
            if measure == 'max':
                diff = features.groupby(by=features.index).mean().values - np.mean(dmso,axis=0).values
                ind_max_diff = np.argmax(np.abs(diff),axis=0)
                parameters[imeas] = diff[ind_max_diff,np.arange(n_feat)]
                #parameters[imeas] = np.max(features.groupby(by=features.index).mean()) - np.mean(dmso,axis=0)
            elif measure == 'mean':
                parameters[imeas] = np.mean(features,axis=0) - np.mean(dmso,axis=0)
            elif measure == 'median':
                parameters[imeas] = np.median(features,axis=0) - np.mean(dmso,axis=0)
            else:
                ValueError('The effect_type can take the values \'max\',\'mean\' or \'median\'. The input value was not recognised.')

            # Apply cutoff
            parameters[imeas][np.abs(parameters[imeas])<=cutoff*np.std(dmso,axis=0)] = 0.0
            parameters[imeas][parameters[imeas]>cutoff*np.std(dmso,axis=0)] = 1.0
            parameters[imeas][parameters[imeas]<-cutoff*np.std(dmso,axis=0)] = -1.0


        parameters = parameters.reshape(1,-1,order='F').flatten()

        self.effect_size_parameters = parameters

        return parameters


    ## Define class methods

    @classmethod
    def add_const_feat(cls,cnstList):

        cls.constant_features += cnstList


if __name__ == '__main__':

    import pandas as pd
    from sklearn.decomposition import PCA

    #d = {'drug_name': ['test','test'], 'drug_dose': ['0','0'], 'col1': [1, 5], 'col2': [0,0], 'col3': [1,2]}
    #df = pd.DataFrame(data=d)

    #d2 = {'drug_name': ['test','test'], 'drug_dose': ['0','0'], 'col1': [0, 0], 'col2': [6,7], 'col3': [1,2]}
    #df2 = pd.DataFrame(data=d2)

    #dr1 = drugClass(df)
    #dr2 = drugClass(df2)

    dat=np.random.rand(5,7)
    d = {'drug_name': ['test']*5, 'drug_dose': ['0']*5,
         'col1': dat[:,0].flatten(), 'col2': dat[:,1].flatten(), 'col3': dat[:,2].flatten(),
         'col4': dat[:,3].flatten(), 'col5': dat[:,4].flatten(), 'col6': dat[:,5].flatten(),
         'col7': dat[:,6].flatten()}
    df = pd.DataFrame(data=d)
    drug = []
    drug.append(drugClass(df))
    drug[0].PCA_fit_transform()
    print('not normalized')
    print(drug[0].Y[:,0:1])

    drug[0].znormalize(updateClass=True)
    drug[0].PCA_fit_transform()
    print('z-normalized')
    print(drug[0].Y[:,0:1])

    pca=PCA()
    Y=pca.fit_transform(drug[0].feat)
    s_sklearn = pca.singular_values_
    print('sklearn z-normalized')
    print(drug[0].Y[:,0:1])

    u,s,v = drug[0].PCA_fit_transform(method='svd')
    print('svd z-normalized')
    Ysvd = np.matmul(drug[0].feat,v.T)
    print(Ysvd[:,0:1])

    print(s_sklearn)
    print(s)

    varexpl_s=[]
    varexpl_e=[]
    sq_s = []
    for i in range(5):
        sq_s.append(s[i]**2)
        varexpl_s.append(s[i]**2/np.sum((s.T).dot(s)))
        varexpl_e.append(drug[0].eigval[i]/np.sum(drug[0].eigval))
    print(varexpl_s)
    print(varexpl_e)

    sq_s = sq_s/np.linalg.norm(sq_s)
    drug[0].eigval = drug[0].eigval/np.linalg.norm(drug[0].eigval)
    print(sq_s)
    print(drug[0].eigval)
