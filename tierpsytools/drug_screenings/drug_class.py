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
import pdb
import numpy as np
import pandas as pd
from warnings import warn

class drugClass():

    nb_of_drugs = 0
    constant_features = []

    ## Initiate class
    def __init__(
            self,
            feat, meta,
            MOAinfo=False,
            set_dose_as_index=False,
            drug_name_col = 'drug_type',
            drug_dose_col = 'drug_dose'
            ):

        ## Define class attributes
        names = meta[drug_name_col].unique()

        if names.shape[0]==1:
            self.drug_name = names[0]
        else:
            raise ValueError('More than one coumpound names found in metadata.')
        self.drug_dose = meta[drug_dose_col].astype(float)

        # If MOAinfo available, read data on compound class/group and mode of action
        if MOAinfo:
            # Read from metadata
            for attr, key in zip(['moa_general', 'moa_specific', 'moa_group'],
                                 ['MOA_general', 'MOA_specific', 'MOA_group']):
                if key in meta:
                    if meta[key].unique().shape>1:
                        pdb.set_trace()
                    self.__dict__[attr] = meta[key].unique()[0]


        # keep feat as a dataframe and set row index = drug dose:
        self.feat = feat.copy()
        if set_dose_as_index:
            self.feat.set_index(meta['drug_dose'], inplace=True)

        self.normalization = None
        self.mil_transform = None

        drugClass.nb_of_drugs += 1

    ## Define class methods
    # Get mean dose points
    def get_mean_doses(self):
        if not hasattr(self, 'mean_doses'):
            self.mean_doses = self.feat.groupby(by=self.drug_dose).mean()
        return self.mean_doses

    def get_dose_dist_from_control(self, control, metric='euclidean'):

        mean_control = np.mean(control, axis=0).reshape(1,-1)
        mean_dose = self.get_mean_doses()

        if 'eucl' in metric:
            from sklearn.metrics import euclidean_distances
            dist = euclidean_distances(mean_control, mean_dose)
        elif 'mahalan' in metric:
            from sklearn.covariance import EmpiricalCovariance
            cov = EmpiricalCovariance().fit(control)
            dist = cov.mahalanobis(mean_dose)

        return dist


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
    def find_const_feat(self):
        """
        FEAT_FILTER_STD: remove features with zero std
        """
        # add the features to the constant feature list of the class (so they can be removed from all classes)
        cnst = self.feat.loc[:,self.feat.var()==0].columns.to_list()
        self.add_const_feat(cnst)
        return self

    def remove_constant_feat(self):
        """
        REMOVE_CONSTANT_FEAT: remove all the features in the class variable constant_features
        """
        self.feat = self.feat[self.feat.columns.difference(self.constant_features)]

    ## Define class methods

    @property
    def moa_info(self):
        return self.moa_general, \
               self.moa_specific, \
               self.moa_group

    @moa_info.setter
    def moa_info(self, value):
        self.moa_general = value[0]
        self.moa_specific = value[1]
        self.moa_group = value[2]

    @property
    def moa_label(self):
        return ' - '.join([self.moa_general, self.moa_specific])

    @moa_label.setter
    def moa_label(self, value):
        self.moa_label = value

    @classmethod
    def add_const_feat(cls,cnstList):

        cls.constant_features += cnstList


class drugClass_v1():

    nb_of_drugs = 0
    constant_features = []

    ## Initiate class
    def __init__(
            self,
            feat, drug_name,
            dose, MOAinfo=None,
            set_dose_as_index=False
            ):

        ## Define class attributes
        self.drug_name = drug_name
        self.drug_dose = dose

        # If MOAinfo available, read data on compound class/group and mode of action
        if MOAinfo is not None:
            self.moa = {}
            if isinstance(MOAinfo, (list, np.ndarray)):
                if np.unique(MOAinfo).shape[0]>1:
                    raise ValueError('MOAinfo not unique.')
                self.moa['MOA_group'] = np.unique(MOAinfo)[0]

            if isinstance(MOAinfo, pd.DataFrame):
                for key in MOAinfo.columns:
                    if MOAinfo[key].unique().shape[0]>1:
                        raise ValueError('MOAinfo not unique.')
                    self.moa[key] = MOAinfo[key].unique()[0]

            if isinstance(MOAinfo, pd.Series):
                if MOAinfo.unique().shape[0]>1:
                    raise ValueError('MOAinfo not unique.')
                self.moa[MOAinfo.name] = MOAinfo.unique()[0]

            if isinstance(MOAinfo, str):
                self.moa['MOA_group'] = MOAinfo

            if isinstance(MOAinfo, dict):
                for key, value in MOAinfo.iteritems():
                    self.moa[key] = value



        # keep feat as a dataframe and set row index = drug dose:
        self.feat = feat.copy()
        if set_dose_as_index:
            self.feat.set_index(dose, inplace=True)

        self.normalization = None
        self.mil_transform = None

        drugClass.nb_of_drugs += 1

    ## Define class methods
    # Get mean dose points
    def get_mean_doses(self):
        if not hasattr(self, 'mean_doses'):
            self.mean_doses = self.feat.groupby(by=self.drug_dose).mean()
        return self.mean_doses

    def get_dose_dist_from_control(self, control, metric='euclidean'):

        mean_control = np.mean(control, axis=0).reshape(1,-1)
        mean_dose = self.get_mean_doses()

        if 'eucl' in metric:
            from sklearn.metrics import euclidean_distances
            dist = euclidean_distances(mean_control, mean_dose)
        elif 'mahalan' in metric:
            from sklearn.covariance import EmpiricalCovariance
            cov = EmpiricalCovariance().fit(control)
            dist = cov.mahalanobis(mean_dose)

        return dist


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
    def find_const_feat(self):
        """
        FEAT_FILTER_STD: remove features with zero std
        """
        # add the features to the constant feature list of the class (so they can be removed from all classes)
        cnst = self.feat.loc[:,self.feat.var()==0].columns.to_list()
        self.add_const_feat(cnst)
        return self

    def remove_constant_feat(self):
        """
        REMOVE_CONSTANT_FEAT: remove all the features in the class variable constant_features
        """
        self.feat = self.feat[self.feat.columns.difference(self.constant_features)]

    ## Define class methods

    @property
    def moa_info(self):
        if not hasattr(self, 'moa'):
            raise Exception ('No MOA info available.')
        return self.moa

    @moa_info.setter
    def moa_info(self, key, value):
        if not hasattr(self, 'moa'):
            self.moa = {}
        self.moa[key] = value

    @property
    def moa_label(self):
        return ' - '.join([self.moa['MOA_general'], self.moa['MOA_specific']])

    @moa_label.setter
    def moa_label(self, value):
        self.moa_label = value

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
