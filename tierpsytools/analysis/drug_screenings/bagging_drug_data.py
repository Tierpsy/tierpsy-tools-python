#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

def shuffle_data(df, arrays, random_state=None):
    """
    Apply the same random shuffling to a dataframe and a list of matching arrays

    Parameters
    ----------
    df : the dataframe
    arrays : list of arrays with size = number of rows of the df
    random_state : random seed

    Returns
    -------
    df : shuffled dataframe
    arrays : shuffled arrays

    """
    df = df.sample(frac=1, replace=False, random_state=random_state)
    for i,a in enumerate(arrays):
        if a is None:
            continue
        arrays[i] = arrays[i][df.index.to_numpy()]

    df = df.reset_index(drop=True)

    return df, arrays

class DrugDataBagging:
    """
    Create bags of drug data by sampling each dose of each drug.
    """
    def __init__(self, n_bags=10, replace=True,
               n_per_dose=None, frac_per_dose=None,
               random_state=None, average_dose=False,
               bluelight_conditions=True):
        """
        n_bags : int
            number of bags to create
        replace : bool
            whether to sample with replacement
        n_per_dose: int or None
            number of samples per dose. If None, frac_per_dose must be defined.
        n_frac_per_dose: int or None
            fraction of dose datapoints to be samples. If None, n_per_dose
            must be defined.
        random_state : int
            random seed
        average_dose : bool
            whether to average the dose samples in each bag
        bluelight_conditions: bool or list of strings
            If the samples are independent of bluelight conditions (bluelight
            stimulus was not used or bluelight conditions are aligned), then
            this parameter must be set to False.
            If each sample has a bluelight label, then this parameter must be set
            to True to sample from each bluelight condition uniformly.
            If True, it is assumed that there are three bluelight labels:
            ['prestim', 'poststim', 'bluelight']. If different labels were used
            or you want to consider a subset of the three conditions, then you
            can give a custom list of bluelight labels.
        """
        self.n_bags = n_bags
        self.replace = replace

        if n_per_dose is None and frac_per_dose is None:
            raise ValueError('Must define either number of samples or ' +
                             'fraction of samples per dose.')
        elif n_per_dose is not None and frac_per_dose is not None:
            raise ValueError('Define either number of samples or ' +
                             'fraction of samples per dose. Cannot '+
                             'use both parameters.')
        self.n = n_per_dose
        self.frac = frac_per_dose
        self.random_state = random_state
        self.average_dose = average_dose
        self._parse_bluelight_conditions(bluelight_conditions)

    def _parse_bluelight_conditions(self, bluelight_conditions):
        if isinstance(bluelight_conditions, bool):
            if bluelight_conditions:
                self.bluelight = ['prestim', 'bluelight', 'poststim']
            else:
                self.bluelight = None
        elif bluelight_conditions is None:
            self.bluelight = None
        elif isinstance(bluelight_conditions, list):
            self.bluelight = bluelight_conditions
        else:
            raise ValueError('Bluelight conditions input type not recognized.')
        return

    def _get_one_bag(self, X, drug, dose, bluelight, random_state=None, shuffle=False):

        if self.bluelight is None:
            Xgrouped = X.groupby(by=[drug, dose])
        else:
            Xgrouped = X.groupby(by=[drug, dose, bluelight])

        Xbag = Xgrouped.apply(
            lambda x: x.sample(n=self.n, frac=self.frac, replace=self.replace,
                               random_state=random_state)
            )

        drug_bag = Xbag.index.get_level_values(0).to_numpy()
        dose_bag = Xbag.index.get_level_values(1).to_numpy()
        if self.bluelight is not None:
            blue_bag = Xbag.index.get_level_values(2).to_numpy()
        else:
            blue_bag = None
        Xbag = Xbag.reset_index(drop=True)

        if self.average_dose:
            if self.bluelight is None:
                Xbag = Xbag.groupby(by=[drug_bag, dose_bag]).agg(np.nanmean)
            else:
                Xbag = Xbag.groupby(by=[drug_bag, dose_bag, blue_bag]).agg(np.nanmean)
                blue_bag = Xbag.index.get_level_values(2).to_numpy()
            drug_bag = Xbag.index.get_level_values(0).to_numpy()
            dose_bag = Xbag.index.get_level_values(1).to_numpy()
            Xbag = Xbag.reset_index(drop=True)

        if shuffle:
            Xbag, [drug_bag, dose_bag, blue_bag] = shuffle_data(
                Xbag, [drug_bag, dose_bag, blue_bag], random_state=self.random_state)

        return Xbag, drug_bag, dose_bag, blue_bag


    def fit(self, X, drug, dose, bluelight=None, shuffle=False):
        """
        Sample the bags from a dataset that contains multiple drugs at multiple doses.

        Parameters
        ----------
        X : dataframe or array
            The tierpsy features per sample.
        drug : array-like
            The drug name for each sample in X
        dose : array-like
            The drug dose for each sample in X
        bluelight : array-like, optional
            The bluelight label for each sample in X. The default is None.
        shuffle : bool, optional
            Whether to shuffle the sampled points in each bag.
            The default is False.

        Returns
        -------
        None.

        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        drug = np.array(drug)
        dose = np.array(dose)
        if bluelight is not None:
            bluelight = np.array(bluelight)

        self.drug_names = np.unique(drug)

        if self.random_state is not None:
            random.seed(self.random_state)

        Xbags = []; drug_bags = []; dose_bags = []; blue_bags = []

        for i in range(self.n_bags):
            if self.random_state is None:
                random_state = None
            else:
                random_state = random.randint(1,1e6)

            Xbag, drug_bag, dose_bag, blue_bag = self._get_one_bag(
                X, drug, dose, bluelight, random_state=random_state,
                shuffle=shuffle)

            Xbags.append(Xbag)
            drug_bags.append(drug_bag)
            dose_bags.append(dose_bag)
            blue_bags.append(blue_bag)


        self.Xbags = Xbags
        self.drug_bags = drug_bags
        self.dose_bags = dose_bags
        if self.bluelight is not None:
            self.blue_bags = blue_bags


    def fit_transform(self, X, drug, dose, bluelight=None, shuffle=False):
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
            raise ValueError('DrugBagging instsnce already fitted.')

        self.fit(X, drug, dose, bluelight=bluelight, shuffle=shuffle)

        meta = []
        for bag in range(len(self.Xbags)):
            metabag = pd.DataFrame({
                'drug_type': self.drug_bags[bag],
                'drug_dose': self.dose_bags[bag],
                })
            if self.bluelight is not None:
                metabag = metabag.assign(bluelight=self.blue_bags[bag])
            meta.append(metabag)

        return self.Xbags, meta

#%%
class SingleDrugBagging(DrugDataBagging):
    def __init__(self, n_bags=10, replace=True,
               n_per_dose=None, frac_per_dose=None,
               random_state=None, average_dose=False,
               bluelight_conditions=True):

        super().__init__(n_bags, replace, n_per_dose, frac_per_dose,
               random_state, average_dose, bluelight_conditions)

    def fit(self, X, dose, bluelight=None, shuffle=False):

        drug = np.ones(X.shape[0])

        super().fit(X, drug, dose, bluelight=bluelight, shuffle=shuffle)

        return

    def fit_transform(self, X, dose, shuffle=False):
        if hasattr(self, 'Xbags'):
            raise ValueError('DrugBagging instance already fitted.')

        self.fit(X, dose, shuffle)

        if self.bluelight is None:
            return self.Xbags, self.dose_bags
        else:
            return self.Xbags, self.dose_bags, self.blue_bags

#%%
class DrugDataBaggingByMOA(DrugDataBagging):
    """
    Get averaged dose samples from a drug screening dataset balancing the
    number of samples per class.
    """
    def __init__(self, multiplier=1, replace=True, n_per_dose=None, frac_per_dose=None,
               random_state=None, bluelight_conditions=True):

        self.multiplier = multiplier
        self.replace = replace

        if n_per_dose is None and frac_per_dose is None:
            raise ValueError('Must define either number of samples or ' +
                             'fraction of samples per dose.')
        elif n_per_dose is not None and frac_per_dose is not None:
            raise ValueError('Define either number of samples or ' +
                             'fraction of samples per dose. Cannot '+
                             'use both parameters.')
        self.n = n_per_dose
        self.frac = frac_per_dose
        self.random_state = random_state

        super()._parse_bluelight_conditions(bluelight_conditions)

    def fit(self, X, moa, drug, dose, bluelight=None, shuffle=False):
        ndrugs_per_moa = pd.DataFrame({'MOA_group':moa, 'drug_type':drug}).groupby(
            by=['MOA_group'])['drug_type'].nunique()
        frac_per_moa = ndrugs_per_moa.max() / ndrugs_per_moa

        mapper = get_drug2moa_mapper(drug, moa)

        self.n_bags = {}

        X_a = {}; meta_a = {}
        for imoa in frac_per_moa.index:
            ind = (moa==imoa)

            n_bags = int(self.multiplier*round(frac_per_moa[imoa]))
            self.n_bags[imoa] = n_bags

            bagger = DrugDataBagging(
                n_bags=n_bags, replace=self.replace, frac_per_dose=self.frac,
                random_state=self.random_state, average_dose=True,
                bluelight_conditions=self.bluelight)

            if bluelight is None:
                X_a[imoa], meta_a[imoa] = bagger.fit_transform(
                        X[ind], drug[ind], dose[ind], shuffle=False)
            else:
                X_a[imoa], meta_a[imoa] = bagger.fit_transform(
                        X[ind], drug[ind], dose[ind], bluelight=bluelight[ind],
                        shuffle=False)

            X_a[imoa] = pd.concat(X_a[imoa], axis=0)
            meta_a[imoa] = pd.concat(meta_a[imoa], axis=0)

        self.X_a = pd.concat(X_a.values())
        self.meta_a = pd.concat(meta_a.values())
        self.meta_a = self.meta_a.assign(MOA_group=pd.Series(self.meta_a['drug_type']).map(mapper))

        return

    def fit_transform(self, X, moa, drug, dose, bluelight=None, shuffle=False):

        if hasattr(self, 'X_a'):
            raise ValueError('DrugBagging instance already fitted.')

        self.fit(X, moa, drug, dose, bluelight, shuffle)

        return self.X_a, self.meta_a


class StrainAugmentDrugData:
    def __init__(self, n_augmented_bags=1, replace=False,
               n_per_dose=None, frac_per_dose=1,
               random_state=None, bluelight_conditions=True):

        self.n_augmented_bags = n_augmented_bags
        self.replace = replace

        if n_per_dose is None and frac_per_dose is None:
            raise ValueError('Must define either number of samples or ' +
                             'fraction of samples per dose.')
        elif n_per_dose is not None and frac_per_dose is not None:
            raise ValueError('Define either number of samples or ' +
                             'fraction of samples per dose. Cannot '+
                             'use both parameters.')
        self.n = n_per_dose
        self.frac = frac_per_dose
        self.random_state = random_state
        if isinstance(bluelight_conditions, bool):
            if bluelight_conditions:
                self.bluelight = ['prestim', 'bluelight', 'poststim']
            else:
                self.bluelight = None
        elif bluelight_conditions is None:
            self.bluelight = None
        elif isinstance(bluelight_conditions, list):
            self.bluelight = bluelight_conditions
        else:
            raise ValueError('Bluelight conditions input type not recognized.')

    def _get_augmented_data(self, Xs, drugs, doses, bluelights):
        Xsample = {}
        for s,X in Xs.items():
            if self.bluelight is None:
                #pdb.set_trace()
                grouped = X.groupby(by=[drugs[s], doses[s]], sort=True)
            else:
                grouped = X.groupby(by=[drugs[s], doses[s], bluelights[s]], sort=True)
            Xsample[s] = grouped.apply(lambda x: x.sample(
                n=self.n, frac=self.frac, replace=self.replace
                ).reset_index(drop=True))
            Xsample[s] = Xsample[s].rename(
                columns={col:'_'.join([str(col),str(s)]) for col in Xsample[s].columns})
        Xaug = pd.concat(list(Xsample.values()), axis=1)
        if Xaug.isna().any().any():
            warnings.warn('There are missing doses or bluelight conditions '+
                          'for some of the drugs in some of the strains.')
        drug_aug = Xaug.index.get_level_values(0).to_numpy()
        dose_aug = Xaug.index.get_level_values(1).to_numpy()
        if self.bluelight is not None:
            blue_aug = Xaug.index.get_level_values(2).to_numpy()
        else:
            blue_aug = None
        Xaug = Xaug.reset_index(drop=True)

        Xaug, [drug_aug, dose_aug, blue_aug] = shuffle_data(
            Xaug, [drug_aug, dose_aug, blue_aug], random_state=self.random_state)

        return Xaug, drug_aug, dose_aug, blue_aug

    def _check_bluelight_input(self, bluelights):
        if not isinstance(bluelights, dict):
            raise ValueError('Expecting dictionary input for bluelights.')
        if self.bluelight is None and bluelights is not None:
            raise ValueError('This StrainAugmentDrugData instance does not '+
                             'accept bluelight conditions.')
        if self.bluelight is not None and bluelights is None:
            raise ValueError('Must give bluelight condition for each sample.')
        if self.bluelight is not None:
            if not all([ all(np.unique(self.bluelight) == np.unique(b))
                        for b in bluelights.values() ]):
                raise Exception('Bluelight conditions missing for some of the groups.')
        return

    def _check_drugs_input(self, drugs):
        drug_names = np.unique(drugs[self.strains[0]])
        for strain in self.strains:
            assert all(drug_names == np.unique(drugs[strain])), \
                'Missing drugs for some of the strains.'
        self.drug_names = drug_names
        return

    def _check_input(self, Xs, drugs, doses, bluelights):
        assert Xs.keys() == drugs.keys() == doses.keys()
        if self.bluelight is not None:
            assert Xs.keys() == bluelights.keys()
        self.strains = list(Xs.keys())
        if self.bluelight is not None:
            self._check_bluelight_input(bluelights)
        self._check_drugs_input(drugs)
        return

    def fit(self, Xs, drugs, doses, bluelights=None, shuffle=False):

        self._check_input(Xs, drugs, doses, bluelights)

        if self.random_state is not None:
            random.seed(self.random_state)

        Xbags = []
        drug_bags = []
        dose_bags = []
        blue_bags = []
        for i in range(self.n_augmented_bags):
            Xaug, drug_aug, dose_aug, blue_aug = \
                self._get_augmented_data(Xs, drugs, doses, bluelights)

            Xbags.append(Xaug)
            drug_bags.append(drug_aug)
            dose_bags.append(dose_aug)
            blue_bags.append(blue_aug)

        self.Xbags = Xbags
        self.drug_bags = drug_bags
        self.dose_bags = dose_bags
        if self.bluelight is not None:
            self.bluelight_bags = blue_bags

    def fit_transform(self, Xs, drugs, doses, bluelights, shuffle=False):
        if hasattr(self, 'Xbags'):
            raise ValueError('DrugBagging instsnce already fitted.')

        self.fit(Xs, drugs, doses, bluelights, shuffle=shuffle)

        if self.bluelight is None:
            if self.n_augmented_bags == 1:
                return self.Xbags[0], self.drug_bags[0], self.dose_bags[0]
            else:
                return self.Xbags, self.drug_bags, self.dose_bags
        else:
            if self.n_augmented_bags == 1:
                return self.Xbags[0], self.drug_bags[0], self.dose_bags[0], self.bluelight_bags[0]
            else:
                return self.Xbags, self.drug_bags, self.dose_bags, self.bluelight_bags


#%%
if __name__=="__main__":
    import matplotlib.pyplot as plt
    n_bags = 4
    n_dose = 3

    X = pd.DataFrame({'a':np.arange(10), 'b':np.arange(10,20)})
    dose = [1,1,1,2,2,2,3,3,3,3]

    single_bagger = SingleDrugBagging(n_bags=n_bags, replace=True,
               n_per_dose=None, frac_per_dose=0.7,
               random_state=None, average_dose=False,
               bluelight_conditions=None)

    Xbags, dose_bags = single_bagger.fit_transform(X, dose)

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

        Xbags, drug_bags, dose_bags = multi_bagger.fit_transform(X, drug, dose, shuffle=True)

        # plt.figure()
        # for i in range(10):
        #     plt.scatter(dose_bags[i], Xbags[i]['a'])

        for i in range(n_bags):
            plt.figure()
            plt.title('Bag {}'.format(i))
            for idrug in np.unique(drug_bags[i]):
                plt.plot(dose_bags[i][drug_bags[i]==idrug], Xbags[i]['a'].values[drug_bags[i]==idrug])


    Xs = {}; doses={}; drugs={}
    for rand_bags in range(3):
        multi_bagger = DrugDataBagging(n_bags=n_bags, replace=True,
                   n_per_dose=None, frac_per_dose=0.7,
                   random_state=10, average_dose=True,
                   bluelight_conditions=None)

        Xbags, drug_bags, dose_bags = multi_bagger.fit_transform(X+rand_bags, drug, dose, shuffle=True)

        Xs[rand_bags] = pd.concat(Xbags, axis=0).reset_index(drop=True)
        doses[rand_bags] = np.concatenate(dose_bags)
        drugs[rand_bags] = np.concatenate(drug_bags)


    augmenter = StrainAugmentDrugData(bluelight_conditions=False)
    Xaug, drug_aug, dose_aug = augmenter.fit_transform(Xs, drugs, doses, shuffle=True)
