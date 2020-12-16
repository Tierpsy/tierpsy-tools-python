#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes that smooth experimental data with multiple replicates per group.

Created on Wed Aug  5 12:55:52 2020

@author: em812
"""
import numpy as np
import pandas as pd
import random
import warnings
import pdb

def get_drug2moa_mapper(drug_id, moa_id):
    """
    Makes disctionary than maps drug names to their MOA
    """
    drug_id = np.array(drug_id)
    moa_id = np.array(moa_id)

    drugs, ind = np.unique(drug_id, return_index=True)
    moas = moa_id[ind]

    return dict(zip(drugs, moas))

class DataBagging:
    """
    Create bags of drug data by sampling from each group.
    """
    def __init__(self, n_bags=10, replace=True,
               n_per_group=None, frac_per_group=None,
               random_state=None, average_sample=False
               ):
        """
        n_bags : int
            number of bags to create
        replace : bool
            whether to sample with replacement
        n_per_group: int or None
            number of samples per group. If None, frac_per_group must be defined.
        frac_per_group: int or None
            fraction of group datapoints to be sampled. If None, n_per_group
            must be defined.
        random_state : int
            random seed
        average_sample : bool
            whether to average the samples within groups in each bag
        """
        self.n_bags = n_bags
        self.replace = replace

        if n_per_group is None and frac_per_group is None:
            raise ValueError('Must define either number of samples or ' +
                             'fraction of samples per group.')
        elif n_per_group is not None and frac_per_group is not None:
            raise ValueError('Define either number of samples or ' +
                             'fraction of samples per group. Cannot '+
                             'use both parameters.')
        self.n = n_per_group
        self.frac = frac_per_group
        self.random_state = random_state
        self.average_sample = average_sample

    def _parse_groupby(self, groupby, X_index):
        """
        Convert groupby parameter to a pandas series objects or a list of
        pandas series.

        """
        dtype_error = 'groupby data type not recognised. The groupby variable(s)'+ \
                      'can be in the form of arrays or pandas Series. To define '+ \
                      'more than one grouping variables, give a list of arrays or '+ \
                      'a list of pandas Series.'

        if groupby is None:
            groupby = pd.Series(np.zeros(X_index.shape[0]), index=X_index, name='group_level_0')
            self.groupby_names_ = [groupby.name]

        elif isinstance(groupby, pd.Series):
            self.groupby_names_ = [groupby.name]

        elif isinstance(groupby, np.ndarray):
            if len(groupby.shape) > 1:
                raise ValueError(dtype_error)
            groupby = pd.Series(groupby, index=X_index, name='group_level_0')
            self.groupby_names_ = [groupby.name]

        elif isinstance(groupby, list):
            names = []
            for i,x in enumerate(groupby):
                if isinstance(x, list):
                    raise ValueError(dtype_error)
                elif isinstance(x, np.ndarray):
                    assert len(x.shape) == 1
                    groupby[i] = pd.Series(x, index=X_index, name='group_level_{}'.format(i))
                    names.append(groupby[i].name)
                elif isinstance(x, pd.Series):
                    names.append(x.name)
                else:
                    raise ValueError(dtype_error)
            self.groupby_names_ = names

        return groupby

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, becase they are all set together
        if hasattr(self, 'Xbags'):
            del self.groupby_names_
            del self.Xbags
            del self.metabags


    def _get_one_bag(self, X, groupby, random_state=None, shuffle=False):
        """
        Sample once from each group

        Parameters
        ----------
        X : pd.DataFrame
            feature matrix
        groupby : pd.Series or list of pd.Series with same index as X
            variable(s) defining groups in X
        random_state : int
            random seed
        shuffle : bool
            Whether to shuffle the sampled datapoints in the returned dataframes

        Returns
        -------
        Xbag : pd.DataFrame
            A dataframe with the sampled data from each group.
        metabag :pd.DataFrame
            A dataframe matching Xbag row-by-row that defines the group label(s)
            for each sample in Xbag.

        """

        n_groupings = len(self.groupby_names_)

        Xgrouped = X.groupby(by=groupby)

        Xbag = Xgrouped.apply(
            lambda x: x.sample(n=self.n, frac=self.frac, replace=self.replace,
                               random_state=random_state)
            )

        if self.average_sample:
            Xbag = Xbag.groupby(level=list(range(n_groupings))).agg(np.nanmean)

        metabag = pd.concat([
            Xbag.index.get_level_values(i).to_frame(index=False)
            for i in range(n_groupings)
            ], axis=1)
        Xbag = Xbag.reset_index(drop=True)

        if shuffle:
            Xbag = Xbag.sample(
                frac=1, replace=False, random_state=self.random_state)
            metabag = metabag.loc[Xbag.index]
            Xbag = Xbag.reset_index(drop=True)
            metabag = metabag.reset_index(drop=True)

        return Xbag, metabag


    def fit(self, X, groupby=None, shuffle=False, refit=False):
        """
        Sample the bags from a dataset that contains multiple groups.

        Parameters
        ----------
        X : dataframe or array
            The feature matrix.
        groupby : pd.Series or list of pd.Series with same index as X
            variable(s) defining groups in X
        random_state : int
            random seed
        shuffle : bool
            Whether to shuffle the sampled datapoints in the returned dataframes.

        Returns
        -------
        None.

        """
        if hasattr(self, 'Xbags'):
            if refit:
                self._reset()
            else:
                raise ValueError('DrugBagging instsnce already fitted.')

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        groupby = self._parse_groupby(groupby, X.index)

        if self.random_state is not None:
            random.seed(self.random_state)

        Xbags = []
        metabags = []

        for i in range(self.n_bags):
            if self.random_state is None:
                random_state = None
            else:
                random_state = random.randint(1,1e6)

            Xbag, metabag = self._get_one_bag(X, groupby, random_state, shuffle)

            Xbags.append(Xbag)
            metabags.append(metabag)

        self.Xbags = Xbags
        self.metabags = metabags


    def fit_transform(self, X, groupby=None, shuffle=False, refit=False):
        """
        Sample the bags from a dataset that contains multiple drugs at multiple doses
        and return them.

        Parameters
        ----------
        Same as fit()

        Raises
        ------
        ValueError
            If the instance has already been fitted.

        Returns
        -------
        list of arrays or dataframes
            The list of bags (sampled data).
        meta : list of dataframes
            The metadata for each bag

        """
        if hasattr(self, 'Xbags'):
            if refit:
                self._reset()
            else:
                raise ValueError('DrugBagging instsnce already fitted.')

        self.fit(X, groupby, shuffle, refit)

        return self.Xbags, self.metabags

#%%
class DataBaggingByClass(DataBagging):
    """
    Get averaged samples from each group of data balancing the
    number of samples per class.
    """
    def __init__(
            self, multiplier=1, replace=True, n_per_group=None,
            frac_per_group=None, balance_classes=False, random_state=None):

        super().__init__(multiplier, replace, n_per_group, frac_per_group,
               random_state, average_sample=True)
        self.multiplier = multiplier
        self.balance = balance_classes

    def _apply_mask(self, groupby, mask):
        if isinstance(groupby, pd.Series):
            return groupby[mask]
        elif isinstance(groupby, list):
            masked_groupby = []
            for x in groupby:
                masked_groupby.append(x[mask])
            return masked_groupby
        else:
            raise ValueError

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, becase they are all set together
        if hasattr(self, 'X_a'):
            del self.X_a
            del self.meta_a
            del self.n_bags

    def fit(self, X, y, groupby=None, shuffle=False, refit=False):

        if hasattr(self, 'X_a'):
            if refit:
                self._reset()
            else:
                raise ValueError('DrugBagging instance already fitted.')

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index, name='class')

        groupby = super()._parse_groupby(groupby, X.index)

        n_per_class = pd.Series(y).value_counts()
        frac_per_class = n_per_class.max() / n_per_class
        if not self.balance:
            frac_per_class[:] = 1

        self.n_bags = {}

        X_a = {}; meta_a = {}
        for iy in frac_per_class.index:
            mask = (y==iy)

            n_bags = int(self.multiplier*round(frac_per_class[iy]))

            self.n_bags[iy] = n_bags

            bagger = DataBagging(
                n_bags=n_bags, replace=self.replace, frac_per_group=self.frac,
                random_state=self.random_state, average_sample=True)

            X_a[iy], meta_a[iy] = bagger.fit_transform(
                        X[mask], groupby=self._apply_mask(groupby, mask),
                        shuffle=False)

            X_a[iy] = pd.concat(X_a[iy], axis=0)
            meta_a[iy] = pd.concat(meta_a[iy], axis=0)
            meta_a[iy][y.name] = iy

        self.X_a = pd.concat(X_a.values()).reset_index(drop=True, inplace=False)
        self.meta_a = pd.concat(meta_a.values()).reset_index(drop=True, inplace=False)

        return

    def fit_transform(self, X, y, groupby=None, shuffle=False, refit=False):

        if hasattr(self, 'X_a'):
            if refit:
                self._reset()
            else:
                raise ValueError('DrugBagging instance already fitted.')

        self.fit(X, y, groupby, shuffle)

        return self.X_a, self.meta_a




#%%
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from tierpsytools.drug_screenings.bagging_drug_data import \
        SingleDrugBagging, DrugDataBagging, DrugDataBaggingByMOA

    # One drug at different doses
    #-------------
    n_bags = 2

    X = pd.DataFrame({'a':np.arange(10), 'b':np.arange(10,20)})
    dose = np.array([1,1,1,2,2,2,3,3,3,3])

    single_bagger = SingleDrugBagging(n_bags=n_bags, replace=True,
               n_per_dose=None, frac_per_dose=0.7,
               random_state=356, average_dose=True,
               bluelight_conditions=None)
    bagger = DataBagging(n_bags=n_bags, replace=True, frac_per_group=0.7,
                         random_state=356, average_sample=True)

    Xbags, dose_bags = single_bagger.fit_transform(X, dose)

    Xbags1, metabags1 = bagger.fit_transform(X, groupby=pd.Series(dose, name='dose'))


    # Three drugs at different doses
    #---------
    X = pd.concat([
        pd.DataFrame({'a':np.arange(10)+i, 'b':np.arange(10,20)})
        for i in range(3)
        ]).reset_index(drop=True)
    dose = np.concatenate([
        [1,1,1,2,2,2,3,3,3,3] for i in range(3)
        ])
    drug = np.concatenate([
        i*np.ones(10) for i in range(3)
        ])


    for rand_bags in range(3):
        multi_bagger = DrugDataBagging(n_bags=n_bags, replace=True,
                   n_per_dose=None, frac_per_dose=0.7,
                   random_state=10, average_dose=True,
                   bluelight_conditions=None)

        bagger = DataBagging(n_bags=n_bags, replace=True, frac_per_group=0.7,
                         random_state=10, average_sample=True)

        Xbags, meta = multi_bagger.fit_transform(X, drug, dose, shuffle=True)
        Xbags1, meta1 = bagger.fit_transform(X, groupby=[pd.Series(drug, name='drug_type'),
                                                         pd.Series(dose, name='drug_dose')],
                                             shuffle=True)

        # plt.figure()
        # for i in range(10):
        #     plt.scatter(dose_bags[i], Xbags[i]['a'])

        # # Plot feature values per bag
        # for i in range(n_bags):
        #     plt.figure()
        #     plt.title('Bag {}'.format(i))
        #     for idrug in np.unique(drug):
        #         mask = meta[i]['drug_type']==idrug
        #         plt.scatter(meta[i].loc[mask, 'drug_dose'], Xbags[i]['a'].values[mask])
        #         plt.xlabel('Drug dose')
        #         plt.ylabel('Feature a')

    # Three drugs at different doses belonging to two classes
    #---------
    X = pd.concat([
        pd.DataFrame({'a':np.arange(10)+i, 'b':np.arange(10,20)})
        for i in range(3)
        ]).reset_index(drop=True)
    dose = np.concatenate([
        [1,1,1,2,2,2,3,3,3,3] for i in range(3)
        ])
    drug = np.concatenate([
        i*np.ones(10) for i in range(3)
        ])
    y = np.concatenate([
        np.zeros(20), np.ones(10)
        ])

    moa_bagger = DrugDataBaggingByMOA(
        multiplier=n_bags, replace=True, n_per_dose=None, frac_per_dose=0.7,
           random_state=245, bluelight_conditions=None)

    bagger = DataBaggingByClass(multiplier=n_bags, replace=True, frac_per_group=0.7,
                     balance_classes=False, random_state=245)

    Xbags, meta = moa_bagger.fit_transform(X, y, drug, dose, shuffle=True)
    Xbags1, meta1 = bagger.fit_transform(
        X, pd.Series(y, name='MOA_group'), groupby=[pd.Series(drug, name='drug_type'), pd.Series(dose, name='drug_dose')],
        shuffle=True)
