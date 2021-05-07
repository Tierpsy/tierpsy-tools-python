#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:00:29 2021

@author: em812
"""
import pandas as pd
import numpy as np
from tierpsytools.analysis.statistical_tests import univariate_tests, get_effect_sizes
from tierpsytools import AUX_FILES_DIR
from pathlib import Path
import matplotlib.pyplot as plt
import pdb

filepath = Path(AUX_FILES_DIR) / 'feature_groups.csv'
default_groups = pd.read_csv(filepath, index_col=0)

class tierpsy_fingerprints():

    def __init__(self, bluelight=True, test='ANOVA', comparison_type='multiclass',
            multitest_method='fdr_by', significance_threshold=0.05,
            test_results=None, groups=None):
        self.bluelight=bluelight
        self.test = test
        self.comparison_type = comparison_type
        self.multitest_method='fdr_by'
        self.p_threshold = significance_threshold
        if groups is None:
            self.groups = default_groups
        else:
            self.groups = self._parse_groups(groups)
        if test_results is not None:
            self.test_results = self._parse_results(test_results)
        return

    def fit(self, X, y, control='N2', n_jobs=-1):
        """
        Get the tierpsy fingerprint:
            Run univariate statistical tests if not provided as input
            Create profile: group and subgroup features and get the stat summaries
                for each group/subgroup

        Parameters
        ----------
        X : pandas dataframe
            features matrix.
        y : pandas series or numpy array
            the labels defining the sample groups to compare (must contain control label).
        control : str, optional
            The control group label. The default is 'N2'.
        n_jobs : int or -1, optional
            number of jobs for parallel processing. Used only if univariate tests
            are ran. The default is -1 (all available cores).

        Returns
        -------
        None.

        """
        if not hasattr(self, 'test_results'):
            self._run_univariate_tests(X, y, control='N2', n_jobs=-1)
        self._create_profile()
        return


    def get_representative_features(self, merge_bluelight=False):
        if not hasattr(self, '_profile'):
            raise ValueError('You must fit the instance first.')

        if not self.bluelight:
            return self._feature_profile(self._profile, bluelight=None)

        if merge_bluelight:
            return self._feature_profile(self.get_profile(merge_bluelight=True), bluelight=None)
        else:
            represent_feat = {}
            for blue,profile in self._profile.items():
                represent_feat[blue] = self._feature_profile(profile, bluelight=blue)
            return represent_feat


    def get_profile(self, feat_set=None, merge_bluelight=False):
        if not hasattr(self, '_profile'):
            raise ValueError('You must fit the instance first.')

        if not self.bluelight:
            return self._profile

        if merge_bluelight:
            return self._get_merged_profile()
        else:
            return self._profile

    def plot_fingerprints(
            self, merge_bluelight, fig=None, ax=None, plot_colorbar=True,
            title=None):

        if merge_bluelight:
            data = self.get_profile(merge_bluelight=merge_bluelight)

            if fig is None and ax is None:
                fig, ax = plt.subplots(figsize=(20,5))
            g = self._plot_one_fingerprint(data, title, ax)

            if plot_colorbar:
                c = g.figure.colorbar(g) #,     fig.colorbar(im, cax=cbar_ax)
                c.set_label('ratio of significant features') #, labelpad=-40, y=1.15, rotation=0)

            plt.tight_layout()
        else:
            profile = self.get_profile(merge_bluelight=merge_bluelight)

            n_plots = len(profile.keys())
            if fig is None and ax is None:
                fig, ax = plt.subplots(n_plots, 1, figsize=(20,5*n_plots))

            for i, (blue, data) in enumerate(profile.items()):
                g = self._plot_one_fingerprint(data, blue, ax[i])

            plt.tight_layout()

            if plot_colorbar:
                fig.subplots_adjust(bottom=0.2)
                cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.008])
                c = plt.colorbar(g, orientation='horizontal', cax=cbar_ax) #  fig.colorbar(im, cax=cbar_ax)
                c.set_label('ratio of significant features') #, labelpad=-40, y=1.15, rotation=0)

        return g

    def _plot_one_fingerprint(self, data, title, ax):
        """
        Plots a signle heatmap with the effect sizes of a set of representative
        features (fingerprint). The set comes from a single bluelight condition
        or the merged profile.

        Parameters
        ----------
        data : pandas dataframe
            the results from a single bluelight condition or the merges profile.
        title : str
            the title of the plot/subplot.
        ax : matplotlib axes object
            the axes object to plot the heatmap in.

        Returns
        -------
        g : seaborn figure object
            the heatmap figure object.

        """
        import seaborn as sns

        data = data[data['best_group']]
        data = data.sort_index(level=0)

        pal = sns.color_palette("Reds", int(data['n_significant'].max())+1) #'Reds'
        xs = np.linspace(0,1,int(data['n_significant'].max())+1)
        pal = {n:c for n,c in zip(xs, pal)}
        lut = data['n_significant_ratio'].apply(lambda x: xs[np.argmin(np.abs(xs-x))])
        lut = lut.map(pal)

        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
        sm.set_array([])

        g= sns.barplot(x=data.index.get_level_values(0), y=data['effect_size_50th'], ax=ax,
                       yerr=data['effect_size_50th_ci'], palette=lut, dodge=False)
        ax.set_title(title)

        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        return g

    def _parse_groups(self, groups):
        if isinstance(groups, pd.DataFrame):
            assert 'group_label' in groups
        elif isinstance(groups, dict):
            df = pd.DataFrame(index=np.concatenate([x for x in groups.values()]))
            for grp, fts in groups.items():
                df.loc[fts] = grp
        return groups

    def _parse_results(self, test_results):
        if not isinstance(test_results, pd.DataFrame):
            raise ValueError('test_results must be a dataframe.')
        else:
            assert np.all(np.isin(['p-value', 'effect_size'], test_results.columns))
        return test_results

    def _feature_profile(self, profile, bluelight=None):

        feature_profile = pd.DataFrame(
            index=profile[profile['best_group']].index.get_level_values(0),
            columns=['feature', 'p-value', 'effect_size'])

        for idx in profile[profile['best_group']].index:
            group, subgroup = idx[0], idx[1]
            if len(idx)==3:
                bluelight=idx[2]
            fts = self._subgroup_features(
                group, subgroup, bluelight=bluelight, feat_set=self.test_results.index)
            res = self.test_results.loc[fts]
            mean = profile.loc[idx, 'effect_size_mean']
            std = profile.loc[idx, 'effect_size_std']
            if np.isnan(std):
                std = 0
            mask = (res['effect_size']>=mean-1.96*std) & (res['effect_size']<=mean+1.96*std)

            feature_profile.loc[group, 'feature'] = res.loc[mask, 'p-value'].idxmin()
            feature_profile.loc[group, ['p-value', 'effect_size']] = res.loc[
                feature_profile.loc[group, 'feature'], ['p-value', 'effect_size']]

        return feature_profile

    def _get_merged_profile(self):
        if not self.bluelight:
            return self._profile

        merged_profile = [x.assign(bluelight=blue)
                          for blue,x in self._profile.items()]
        merged_profile = pd.concat(
            [x.set_index('bluelight', append=True) for x in merged_profile],
            axis=0)

        merged_profile = self._mark_best_subgroup(merged_profile)
        return merged_profile

    def _run_univariate_tests(self, X, y, control='N2', n_jobs=-1):

        stats, pvals, _ = univariate_tests(
            X, y, control=control, test=self.test,
            comparison_type=self.comparison_type,
            multitest_correction=self.multitest_method,
            n_jobs=n_jobs)

        effects = get_effect_sizes(
            X, y, control=control, test=self.test,
            comparison_type=self.comparison_type)

        test_res = pd.DataFrame(pvals.min(axis=1), columns=['p-value'])

        # In most cases, the pvals and effects have the same shape
        # (when we do group-by-group comparisons, we get group-by-group
        # effect sizes too, and when we do multi-class comparisons we get one
        # effect size).
        # But for the Kruskal-Wallis case, we cannot get one effect size for the
        # test, so we get group-by-group effect sizes instead and keep the max.
        # In this case pvals has only one column, but effects has more than one
        # columns
        if pvals.shape==effects.shape:
            test_res['effect_size'] = effects.values[pvals.isin(pvals.min(axis=1)).values]
        else:
            test_res['effect_size'] = effects.max(axis=1)

        self.test_results = test_res

        return

    def _profile_info(self, res, groupby, n_boot=1000):
        """
        Gets stat summaries for effect sizes and number of significant features
        for each subgroup and stores them in the profile dataframe.

        Parameters
        ----------
        res : pandas dataframe
            Dataframe with the results of the univariate statistical tests
            (p-values and effect sizes). It also includes the group and subgroup
            labels for each feature.
        groupby : list of strings
            The group and subgroup labels and the bluelight label if applicable.
        n_boot : int, optional
            number of bootstrap samples for the stats estimates. The default is 1000.

        Returns
        -------
        profile : pandas dataframe
            summary stats of effect sizes and number of significant features
            per features subgroup.

        """
        from tierpsytools.analysis.statistical_tests import bootstrapped_ci as boot_ci
        grouped_res = res.groupby(by=groupby)

        profile = grouped_res.agg(
            effect_size_mean= ('effect_size', lambda x: np.mean(x)),
            effect_size_mean_ci = ('effect_size', lambda x: np.diff(boot_ci(x, np.mean, n_boot))/2),
            effect_size_10th = ('effect_size', lambda x: np.quantile(x, 0.1)),
            effect_size_50th= ('effect_size', lambda x: np.median(x)),
            effect_size_50th_ci = ('effect_size', lambda x: np.diff(boot_ci(x, np.median, n_boot))/2),
            effect_size_90th= ('effect_size', lambda x: np.quantile(x, 0.9)),
            effect_size_std= ('effect_size', lambda x: np.std(x)),
            n_significant = ('p-value', lambda x: (x<self.p_threshold).sum()),
            n_significant_ratio = ('p-value', lambda x: (x<self.p_threshold).sum()/x.shape[0])
            )

        profile = self._mark_best_subgroup(profile)

        return profile

    def _mark_best_subgroup(self, profile):
        if 'best_group' in profile:
            profile.drop(columns=['best_group'], inplace=True)

        best_group = profile['n_significant'].groupby(level=0).idxmax()
        profile = profile.assign(best_group=False)
        profile.loc[best_group, 'best_group'] = True
        return profile


    def _create_profile(self):
        """
        Get group-subgroup stats and select most important subgroup
        """
        if not hasattr(self, 'test_results'):
            raise ValueError('First you must run the univariate tests.')

        if self.bluelight:
            blue_label = [ft.split('_')[-1] for ft in self.test_results.index]

            profile = {}
            for blue,res in self.test_results.groupby(by=blue_label):
                fts = ['_'.join(ft.split('_')[:-1]) for ft in res.index]
                groupby = [self.groups.loc[fts,'group_label'].values,
                           self.groups.loc[fts,'motion_label'].values]

                profile[blue] = self._profile_info(res, groupby)
        else:
            fts = self.test_results.index.to_list()
            groupby = [self.groups.loc[fts,'group_label'],
                       self.groups.loc[fts,'motion_label']]

            profile = self._profile_info(self.test_results, groupby)

        self._profile=profile

        return profile

    def _group_features(self, group, bluelight=None, feat_set=None):

        fts = self.groups[self.groups['group_label']==group].index.to_list()

        if bluelight is not None:
            if bluelight not in ['prestim', 'bluelight', 'poststim']:
                raise ValueError('Bluelight condition not recognised.')
            fts = ['_'.join([ft, bluelight]) for ft in fts]

        if feat_set is not None:
            fts = [ft for ft in fts if ft in feat_set]
        return fts

    def _subgroup_features(self, group, subgroup, bluelight=None, feat_set=None,
                           sub_label='motion_label'):
        fts = self.groups[(self.groups['group_label']==group) &
                      (self.groups[sub_label]==subgroup)].index.to_list()

        if bluelight is not None:
            if bluelight not in ['prestim', 'bluelight', 'poststim']:
                raise ValueError('Bluelight condition not recognised.')
            fts =  ['_'.join([ft, bluelight]) for ft in fts]

        if feat_set is not None:
            fts = [ft for ft in fts if ft in feat_set]
        return fts

