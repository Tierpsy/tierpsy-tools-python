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

groups_datatype =  \
    """The groups can be defined in the form of a dataframe with feature names as index:
              | group_label | subgroup_label | ...
    ------------------------------------------
    feature_1 | group1      | subgroup1      |
    feature_2 | group1      | subgroup2      |
    feature_3 | group2      | subgroup1      |
    feature_4 | group2      | subgroup1      |
    ....      | ...         | ...            |
    or in the form of a dictionary:
    {group1: [feature_1, feature_3, ...]
     group2: [feature_2, ...],
     ...} """

test_results_datatype =  \
    """The test results must be defined in the form of a dataframe with feature names as index:
              | p-value   | effect_size    |
    ---------------------------------------
    feature_1 | p-val_1   | effect_size_1  |
    feature_2 | p-val_2   | effect_size_2  |
    ....      | ...       | ...            |
    """

class tierpsy_fingerprints():

    def __init__(self, bluelight=True, test='ANOVA', comparison_type='multiclass',
            multitest_method='fdr_by', significance_threshold=0.05,
            test_results=None, groups=None, groupby=['group_label', 'motion_label'],
            representative_feat=None):
        """
        Class definition

        Parameters
        ----------
        bluelight : bool, optional
            Whether there are bluelight conditions in the experiment. The default is True.
        test : str, optional - ignored if the test_results are given as input
            Type of statistical test to perform if the test_results are not
            given as input. The options are the ones available in
            tierpsytools.analysis.statistical_tests.univariate_tests
            ('ANOVA', 'Kruskal-Wallis', 'Mann-Whitney test' or 't-test')
            The default is 'ANOVA'.
        comparison_type : str, optional - ignored if the test_results are given as input
            The type of comparison to make. See the docstring of
            tierpsytools.analysis.statistical_tests.univariate_tests for the
            definition of the parameter.
            The default is 'multiclass'.
        multitest_method : string or None, optional - ignored if the test_results are given as input
            Method to use to correct p-values for multiple comparisons.
            The options are the ones available in statsmodels.stats.multitest.multipletests.
            The default is 'fdr_by'.
        significance_threshold : float, optional
            The significance threshold for the p-values. The default is 0.05.
        test_results : dataframe or None, optional
            If statistical tests have been performed in advance, then the results
            can be given as input to the class object in the form of a dataframe
            with feature names as index:
                          | p-value   | effect_size    |
                ---------------------------------------
                feature_1 | p-val_1   | effect_size_1  |
                feature_2 | p-val_2   | effect_size_2  |
                ....      | ...       | ...            |

            If None, then statistical tests will be performed when the fit method
            is called.
            The default is None.
        groups : dataframe or dict or None, optional
            If None, the features will be grouped in 98 groups based on their
            names and subgrouped based on their motion mode (default).
            Alternatively, custom groups can be defined in two ways:
                - In the form of a dataframe with feature names as index:
                              | group_label | subgroup_label | ...
                    ------------------------------------------
                    feature_1 | group1      | subgroup1      |
                    feature_2 | group1      | subgroup2      |
                    feature_3 | group2      | subgroup1      |
                    feature_4 | group2      | subgroup1      |
                    ....      | ...         | ...            |

                - In the form of a dictionary (only if there are no subgroups):
                    { group1: [feature_1, feature_3, ...],
                      group2: [feature_2, ...],
                      ... }
                  The dictionary is parsed to a dataframe with a "group_label" column.
            The default is None.
        groupby: list of column names, optional
            The list of the columns names in the groups dataframe that will be
            used to group the features.
            The default is ['group_label', 'motion_label'].
        representative_feat : column name or None, optional
            If None, the representative feature for each group/subgroup are
            not predefined, but instead selected automatically based on the test
            results.
            The user can pre-define a representative feature for each group/subgroup
            with an additional column in the groups dataframe:
                              | group_label | subgroup_label | representative_feature |
                    ------------------------------------------------------------------
                    feature_1 | group1      | subgroup1      | True
                    feature_2 | group1      | subgroup2      | True
                    feature_3 | group2      | subgroup1      | False
                    feature_4 | group2      | subgroup1      | True
                    ....      | ...         | ...            | ...
            In this case the parameter representative_feat is the name of the
            column that defines the representative feature.
            The default is None.

        """
        self.bluelight=bluelight
        self.test = test
        self.comparison_type = comparison_type
        self.multitest_method='fdr_by'
        self.p_threshold = significance_threshold
        if groups is None:
            self.groups = default_groups
        else:
            self.groups = self._parse_groups(groups)
        self._parse_groupby(groupby)
        self._parse_representative(representative_feat)
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
        """
        Get a set of representative features (one feature from each group).
        The representative features can be predefined during the creation of
        the object together with the definition of the groups. If they have not
        been predefined, they are selected based on the test_results.

        If there are subgroups, then the representative feature for each
        group is selected from the best subgroup.

        The most representative feature of a group/subgroup is the feature
        with the min p-value within a range of effect sizes +-2*std from the mean
        effect size of the group.

        Parameters
        ----------
        merge_bluelight : bool, optional
            If there are bluelight conditions, this parameter defines whether
            we will pick representative features separately for each bluelight
            condition or pick them from the merged profile. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        represent_feat : dataframe or dict with dataframes for each blue condition
            Datagrame containing the representative features with their respective
            p-values and effect sizes.

        """
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


    def get_profile(self, merge_bluelight=False):
        """
        Get the result stats per group and subgroup (profile) and the
        best subgroup selection (the subgroup with the largest number of
                                 significant features).
        If there are multiple bluelight conditions, then there is the option
        to get a separate profile per bluelight condition or to get a single
        merged profile, where the best subgroup is chosen among all the bluelight
        conditions.

        Parameters
        ----------
        merge_bluelight : bool, optional
            Whether to select the best subgroup among all bluelight conditions
            (merge_bluelight=True) or to select separately for each bluelight
            condition (merge_bluelight=False).
            If there are no bluelight conditions (self.bluelight is False) then
            this parameter is ignored.
            The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        profile: dataframe or dict
            The profile (results stats per group and subgroup) is returned in the
            form of a dataframe. If there are multiple bluelight conditions and
            merge_bluelight=False, then the function returns a dictionary with
            a profile per bluelight condition.

        """
        if not hasattr(self, '_profile'):
            raise ValueError('You must fit the instance first.')

        if not self.bluelight:
            return self._profile

        if merge_bluelight:
            return self._get_merged_profile()
        else:
            return self._profile

    def plot_fingerprints(
            self, merge_bluelight=False, fig=None, ax=None, plot_colorbar=True,
            title=None, feature_names_as_xticks=False, saveto=None):
        """
        Plot barplots with the stasts of the effect size and the number of
        significant features per feature group.

        Parameters
        ----------
        merge_bluelight : bool
            If True, one barplot will be plotted. For each feature group, the
            bluelight condition that wields more significant differences will be
            selected and plotted.
            If False, three barplots will be plotted, one for each bluelight condition.
            If there is no bluelight stimulus in the data, then this parameter is ignored.
        fig : matplotlib figure object, optional
            The user can give a figure object in which the fingerprint(s) will be plotted.
            The default is None.
        ax : matplotlib axes object, optional
            The user can give the axes in which the fingerprint(s) will be plotted.
            The default is None.
        plot_colorbar : bool, optional
            If True, the colorbar for the ratio of significant feature in each group
            will be added to the figure.
            The default is True.
        title : string, optional
            The user can define a title for the plot.
            The default is None.
        feature_names_as_xticks : bool, optional
            IF False, the xticks of the barplot will be the group labels.
            If True, the xticks of the barplot will be the names of the representative
            features per group.
            The default is False.
        saveto : path, optional
            The path to save the figure (including file name).

        Returns
        -------
        fig : maplotlib figure object
            The entire figure object is returned.

        """
        if merge_bluelight or self.bluelight is False:
            data = self.get_profile(merge_bluelight=merge_bluelight)

            if feature_names_as_xticks:
                xticklabels = self._get_ft_xticklabels(data, merge_bluelight)
            else:
                xticklabels = data.index.to_list()

            if fig is None and ax is None:
                fig, ax = plt.subplots(figsize=(20,5))

            g,sm = self._plot_one_fingerprint(data, title, ax, xticklabels)

            if plot_colorbar:
                c = g.figure.colorbar(sm) #,     fig.colorbar(im, cax=cbar_ax)
                c.set_label('ratio of significant features') #, labelpad=-40, y=1.15, rotation=0)

            plt.tight_layout()
        else:
            profile = self.get_profile(merge_bluelight=merge_bluelight)

            n_plots = len(profile.keys())
            if fig is None and ax is None:
                fig, ax = plt.subplots(n_plots, 1, figsize=(20,5*n_plots))

            for i, (blue, data) in enumerate(profile.items()):

                if feature_names_as_xticks:
                    xticklabels = self._get_ft_xticklabels(data, merge_bluelight)
                else:
                    xticklabels = data.index.to_list()

                if i==n_plots-1:
                    g,sm = self._plot_one_fingerprint(data, blue, ax[i], xticklabels)
                else:
                    g,sm = self._plot_one_fingerprint(data, blue, ax[i], xticklabels, xticks=False)

            plt.tight_layout()

            if plot_colorbar:
                fig.subplots_adjust(bottom=0.2)
                cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.008])
                c = plt.colorbar(sm, orientation='horizontal', cax=cbar_ax) #  fig.colorbar(im, cax=cbar_ax)
                c.set_label('ratio of significant features') #, labelpad=-40, y=1.15, rotation=0)

        if saveto is not None:
            plt.savefig(saveto)

        return g

    def plot_boxplots(self, X, y, saveto, control='N2'):
        """
        Plot boxplots of the representative features from each group.
        Since the fingerprint class object does not store all the feature information,
        the user needs to give the feature matrix as input to this function.

        Parameters
        ----------
        X : dataframe, shape (n_samples, n_features)
            The feature matrix.
        y : array-like, shape (n_samples,)
            The labels for each sample.
        saveto : path
            The path to save all the plots (not including file name).
        control : str, optional
            The label in y that corresponds to the controls. The default is 'N2'.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        import seaborn as sns

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = np.array(y)

        if not np.isin(control, y):
            raise Exception('The control label is not found in y.')
        if np.unique(y).shape[0]<2:
            raise Exception('Less than two unique groups found in y.')

        saveto = Path(saveto)

        fts = self.get_representative_features(merge_bluelight=True)['feature'].values

        for ft in fts:
            if not ft in X.columns:
                raise Exception('At least one of the representative features ' +
                                'is not in the features dataframe given as input.')
            fig, ax = plt.subplots()
            g = sns.boxplot(y=X[ft], x=y, ax=ax, order=[control]+list(np.unique(y[y!=control])))
            plt.savefig(saveto/ft, dpi=300)
            plt.close()
        return


    def _plot_one_fingerprint(self, data, title, ax, xticklabels, xticks=True):
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
        xticks : bool
            whether the xticks will be included in the barplot or not
            (the xticks are either the group labels or the representative feature
             names when feature_names_as_xticks=True)

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
        if xticks:
            g.set(xticklabels=xticklabels)
        else:
            g.set(xticklabels=[])
        ax.set_title(title)

        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        return g, sm

    def _get_ft_xticklabels(self, data, merge_bluelight):
        """
        Change the index from group id to representative feature name.
        """
        if hasattr(self, 'representative_feats'):
            xticklabels = data.index.get_level_values(0).map(
                dict(self.representative_feats.reset_index(drop=False).values))
        else:
            rf = self.get_representative_features(merge_bluelight=merge_bluelight)
            xticklabels = data.index.get_level_values(0).map(
                dict(rf[['feature']].reset_index(drop=False).values)).to_list()
        return xticklabels

    def _parse_groups(self, groups):
        """
        Parse groups to dataframe format

        Parameters
        ----------
        groups : dataframe or dictionary
            DESCRIPTION.

        Returns
        -------
        groups : TYPE
            DESCRIPTION.

        """
        if isinstance(groups, pd.DataFrame):
            pass
        elif isinstance(groups, dict):
            df = pd.DataFrame(
                index=np.concatenate([x for x in groups.values()]),
                columns=['group_label'])
            for grp, fts in groups.items():
                df.loc[fts, 'group_label'] = grp
        else:
            raise Exception(groups_datatype)
        return groups

    def _parse_results(self, test_results):
        if not isinstance(test_results, pd.DataFrame):
            raise Exception(test_results_datatype)
        else:
            assert np.all(np.isin(['p-value', 'effect_size'], test_results.columns)), \
                test_results_datatype
        return test_results

    def _parse_groupby(self, groupby):
        if not isinstance(groupby, (list, np.ndarray)):
            raise Exception('groupby must be a list of columns names.')
        for col in groupby:
            if not col in self.groups.columns:
                raise Exception(f'There is no column named \'{col}\' in groups.')
        self.groupby = groupby

    def _parse_representative(self, representative_feat):
        """
        If the representstive feature for each group is predefined, this
        function returns a dataframe with the group/subgroup as index and
        the corresponding representative feature in a 'feature' column.
        """
        if representative_feat is None:
            return

        if not isinstance(representative_feat, str):
            raise Exception(
                'The parameter representative_feat can be either None or' +
                'the name of a column in groups. Data type not recognised.')

        if not representative_feat in self.groups.columns:
            raise Exception(
                f'There is no column called {representative_feat} in the' +
                'groups dataframe.')

        # get the groups info for the representative features
        fts = self.groups.loc[self.groups[representative_feat], self.groupby]
        # move the feature name from index to a column
        fts.index = fts.index.set_names(['feature'])
        fts = fts.reset_index(drop=False)
        # set the group ids as index
        fts = fts.set_index(self.groupby, drop=True)

        self.representative_feats = fts


    def _feature_profile(self, profile, bluelight=None):
        """
        Get the feature profile (representative features from each group)
        for a specific bluelight condition (if applicable).
        If the feature profile has been predefined during the creation of
        the object, then the predefined features will be returned.
        If the feature profile has not been predefined, then the function will
        find the most representative feature from the best subgroup of each
        group. The most representative feature in a subgroup is a feature
        i) with an effect size within +-2*std from the mean of the subgroup
        ii) which has the smallest p-value within this range of effect sizes


        Parameters
        ----------
        profile : dataframe
            The summary results per group and per motion subgroup.
            If there are different bluelight conditions, then the profile must
            refer to one of the conditions (specified in the bluelight parameter)
            or be the merged profile (in this case the bluelight parameter is None).
        bluelight : str or None, optional
            The bluelight condition from which to select features.
            None if there is no bluelight stimulus or if the profile is merged.
            The default is None.

        Returns
        -------
        feature_profile : dataframe
            The representative features with their corresponding
            p-values and effect sizes.

        """
        if hasattr(self, 'representative_feats'):
            return self._predefined_feature_profile(profile, bluelight)
        else:
            return self._selected_feature_profile(profile, bluelight)


    def _predefined_feature_profile(self, profile, bluelight):
        """
        Creat the feature profile based on pre-defined representative features
        per group.
        """

        # get the best subgroups (if applicable - otherwise all groups are selected)
        if 'bluelight' in profile.index.names:
            profile = profile.reset_index(level='bluelight')
        groupids = profile[profile['best_group']].index

        # get the representative feature for each selected group/subgroup
        feature_profile = self.representative_feats.loc[groupids]

        # add bluelight label, if applicable
        if bluelight is not None:
            feature_profile['feature'] = feature_profile['feature'] + f'_{bluelight}'
        elif 'bluelight' in profile.columns:
            bluelabels = profile.loc[profile['best_group'], 'bluelight']
            feature_profile['feature'] = \
                feature_profile['feature'].str.cat(bluelabels, sep='_')

        # keep only the primary group as index
        feature_profile = feature_profile.set_index(
            feature_profile.index.get_level_values(0), drop=True)

        # get the test results for these features
        feature_profile['p-value'] = \
            self.test_results.loc[feature_profile['feature'].values, 'p-value'].values
        feature_profile['effect_size'] = \
            self.test_results.loc[feature_profile['feature'].values, 'effect_size'].values

        return feature_profile

    def _selected_feature_profile(self, profile, bluelight):
        """
        Select the representative features based on the test results of a
        specific profile.
        """
        feature_profile = pd.DataFrame(
            index=profile[profile['best_group']].index.get_level_values(0),
            columns=['feature', 'p-value', 'effect_size'])

        for idx in profile[profile['best_group']].index:
            if isinstance(idx, tuple):
                groupid = {key: value for key,value in zip(self.groupby, idx[:len(self.groupby)])}
                if len(idx)==len(self.groupby)+1:
                    bluelight=idx[-1]
            else:
                groupid = {self.groupby[0]: idx}

            fts = self._get_features_group(
                groupid, bluelight=bluelight, feat_set=self.test_results.index)
            res = self.test_results.loc[fts]
            mean = profile.loc[idx, 'effect_size_mean']
            std = profile.loc[idx, 'effect_size_std']
            if np.isnan(std):
                std = 0
            mask = (res['effect_size']>=mean-1.96*std) & (res['effect_size']<=mean+1.96*std)

            # get the primary group
            group = groupid[self.groupby[0]]

            # store the representative feature of the group
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
            X, y, control=control, linked_test=self.test,
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
            min_pvals = pvals.min(axis=1)
            nan_mask = pvals.min(axis=1).isna().values
            min_pval_mask = pvals.isin(min_pvals).values
            min_pval_mask[nan_mask, 0] = True
            test_res['effect_size'] = effects.values[min_pval_mask]
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
                groupby = [*self.groups.loc[fts, self.groupby].values.T]

                profile[blue] = self._profile_info(res, groupby)
        else:
            fts = self.test_results.index.to_list()
            groupby = [*self.groups.loc[fts, self.groupby].values.T]

            profile = self._profile_info(self.test_results, groupby)

        self._profile=profile

        return profile

    def _get_features_group(self, group, bluelight=None, feat_set=None):
        """
        Helper function to get all the features of a given group

        Parameters
        ----------
        group : dict
            a dictionary of the form :
                {group_label_0: group_to_select, group_label_1: group_to_select, ...}
            where group_label is the column in the self.groups dataframe
            that defines the group and group_to_select is a
            group name appearing in the values of this column.
        bluelight : str or None, optional
            The bluelight condition to append to the feature names.
            The default is None.
        feat_set : list of feature names or None, optional
            If it is not None, then the function will return only features
            belonging to this specific feature set.
            The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        fts : list of feature names
            The list of features belonging to the selected group.

        """
        from functools import reduce

        masks = [(self.groups[group_label]==group_to_select).values
                 for group_label, group_to_select in group.items()]
        masks = reduce(lambda x,y: x & y, masks)
        fts = self.groups[masks].index.to_list()

        if bluelight is not None:
            if bluelight not in ['prestim', 'bluelight', 'poststim']:
                raise ValueError('Bluelight condition not recognised.')
            fts = ['_'.join([ft, bluelight]) for ft in fts]

        if feat_set is not None:
            fts = [ft for ft in fts if ft in feat_set]
        return fts

    # def _subgroup_features(self, group, subgroup, bluelight=None, feat_set=None,
    #                        sub_label='motion_label'):
    #     fts = self.groups[(self.groups['group_label']==group) &
    #                   (self.groups[sub_label]==subgroup)].index.to_list()

    #     if bluelight is not None:
    #         if bluelight not in ['prestim', 'bluelight', 'poststim']:
    #             raise ValueError('Bluelight condition not recognised.')
    #         fts =  ['_'.join([ft, bluelight]) for ft in fts]

    #     if feat_set is not None:
    #         fts = [ft for ft in fts if ft in feat_set]
    #     return fts

