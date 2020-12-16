#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 17:22:40 2020

@author: em812
"""

def l2_stack_majority_vote_CV(
        Xstr, ystr, groupstr,
        base_estimator, stacked_estimator,
        splitter, stacked_splitter,
        vote_type='counts', l1_pred_type='counts',
        l2_pred_type='single_pred',
        scale_function=None,
        stacked_scale_function=None,
        retrain_estimators=False,
        n_jobs=-1, scorer=None
        ):
    """
    Level 2:
        Single compound MOA classification
    Level 1:
        Strains
    Level 0:
        Bluelight conditions per strain

    Bluelight conditions --> majority vote --> compound-level predictions

    Stacked bluelight conditions, Y_group ---> Strain-specific compound-level predictions

    Stacked strain-level predictions, Y_group --> Final compound-level predictions
    """

    l1_scores = {key: {k:{} for k in Xstr[key].keys()} for key in Xstr.keys()}
    l2_scores = {}

    l2_y = None
    l2_predictions = []
    for (key2, Xs), (key2, ys), (key2, groups) in zip(Xstr.items(), ystr.items(), groupstr.items()):
        l1_y = None
        l1_predictions = []
        for (key1,X), (key1,y), (key1,group) in zip(Xs.items(),ys.items(),groups.items()):
            node = StackNode(base_estimator, splitter, is_grouped_data=True,
                             vote_type=vote_type, scale_function=scale_function,
                             n_jobs=-1, scorer=scorer)
            targ, pred = node.fit_transform(X, y, group, pred_type=l1_pred_type)

            if l1_y is None:
                l1_y = targ
            else:
                assert all(l1_y == targ)
            l1_predictions.append(pred)

            l1_scores[key2][key1]['standard'] = node.scores
            l1_scores[key2][key1]['majority_vote'] = node.scores_group

        l1_predictions = pd.concat(l1_predictions, axis=1)

        node = StackNode(stacked_estimator, stacked_splitter, is_grouped_data=False,
                         scale_function=stacked_scale_function, n_jobs=n_jobs,
                         scorer=scorer)
        targ, pred = node.fit_transform(l1_predictions, l1_y, pred_type=l2_pred_type)

        if l2_y is None:
            l2_y = targ
        else:
            assert all(l2_y==targ)
        l2_predictions.append(pred)

        l2_scores[key2] = node.scores

    l2_predictions = np.concatenate(l2_predictions, axis=1)

    node = StackNode(stacked_estimator, stacked_splitter, is_grouped_data=False,
                    scale_function=stacked_scale_function, n_jobs=n_jobs,
                    scorer=scorer)

    node.fit(l2_predictions, l2_y)

    return l1_scores, l2_scores, node.scores


def first_stack_then_majority_vote_CV(
        Xstr, ystr, groupstr,
        base_estimator, stacked_estimator,
        splitter, stacked_splitter,
        vote_type='counts', l0_pred_type='probas',
        l1_pred_type='counts',
        scale_function=None,
        stacked_scale_function=None,
        retrain_estimators=False,
        n_jobs=-1, scorer=None
        ):

    """
    Level 2:
        Single compound MOA classification
    Level 1:
        Strains
    Level 0:
        Bluelight conditions per strain

    Bluelight conditions --> well-level predictions

    Stacked bluelight conditions, y ---> Strain-specific well-level predictions -->
        majority_vot --> Strain-specific compound-level predictions

    Stacked strain-level predictions, Y_group --> Final compound-level predictions
    """

    l0_scores = {key: {k:{} for k in Xstr[key].keys()} for key in Xstr.keys()}
    l1_scores = {key: {} for key in Xstr.keys()}

    l1_y = None
    l1_predictions = []
    for (key1, Xs), (key1, ys), (key1, groups) in zip(Xstr.items(), ystr.items(), groupstr.items()):
        l0_y = None
        l0_predictions = []
        for (key0,X), (key0,y), (key0,group) in zip(Xs.items(),ys.items(),groups.items()):
            node = StackNode(base_estimator, splitter, is_grouped_data=True,
                             vote_type=vote_type, scale_function=scale_function,
                             n_jobs=-1, scorer=scorer)
            targ, pred = node.fit_transform(X, y, group, pred_type=l0_pred_type, group_pred=False)

            if l0_y is None:
                l0_y = targ
            else:
                assert all(l0_y == targ)
            l0_predictions.append(pred)

            l0_scores[key1][key0]['standard'] = node.scores
            l0_scores[key1][key0]['majority_vote'] = node.scores_group

        l0_predictions = np.concatenate(l0_predictions, axis=1)

        node = StackNode(stacked_estimator, stacked_splitter, is_grouped_data=True,
                         scale_function=stacked_scale_function, n_jobs=n_jobs,
                         scorer=scorer, vote_type=vote_type, )
        targ, pred = node.fit_transform(l0_predictions, l0_y, group=group,
                                        pred_type=l1_pred_type, group_pred=True)

        if l1_y is None:
            l1_y = targ
        else:
            assert all(l1_y==targ)
        l1_predictions.append(pred)

        l1_scores[key1]['standard'] = node.scores
        l1_scores[key1]['majority_vote'] = node.scores_group

    l1_predictions = np.concatenate(l1_predictions, axis=1)

    node = StackNode(stacked_estimator, stacked_splitter, is_grouped_data=False,
                    scale_function=stacked_scale_function, n_jobs=n_jobs,
                    scorer=scorer)

    node.fit(l1_predictions, l1_y)

    return l0_scores, l1_scores, node.scores


def l2_stack_then_majority_vote_CV(
        Xstr, ystr, groupstr,
        base_estimator, stacked_estimator,
        splitter, stacked_splitter,
        vote_type='counts', l0_pred_type='probas',
        l1_pred_type='counts',
        scale_function=None,
        stacked_scale_function=None,
        retrain_estimators=False,
        n_jobs=-1, scorer=None
        ):

    """
    Two-level VERTICAL STACKING V1

    Level 2:
        Single compound MOA classification
    Level 1:
        Strains
    Level 0:
        Bluelight conditions per strain

    Bluelight conditions --> well-level predictions

    Stacked bluelight conditions, y ---> Strain-specific well-level predictions

    Vertically Stacked well-level predictions, y* --> Final well-level predictions -->
        majority_vote --> final compound-level predictions
    """

    l0_scores = {key: {k:{} for k in Xstr[key].keys()} for key in Xstr.keys()}
    l1_scores = {key: {} for key in Xstr.keys()}

    l1_group = []
    l1_y = []
    l1_predictions = []
    for (key1, Xs), (key1, ys), (key1, groups) in zip(Xstr.items(), ystr.items(), groupstr.items()):
        l0_y = None
        l0_predictions = []
        for (key0,X), (key0,y), (key0,group) in zip(Xs.items(),ys.items(),groups.items()):
            node = StackNode(base_estimator, splitter, is_grouped_data=True,
                             vote_type=vote_type, scale_function=scale_function,
                             n_jobs=-1, scorer=scorer)
            targ, pred = node.fit_transform(X, y, group, pred_type=l0_pred_type, group_pred=False)

            if l0_y is None:
                l0_y = targ
            else:
                assert all(l0_y == targ)
            l0_predictions.append(pred)

            l0_scores[key1][key0]['standard'] = node.scores
            l0_scores[key1][key0]['majority_vote'] = node.scores_group

        l0_predictions = np.concatenate(l0_predictions, axis=1)

        node = StackNode(stacked_estimator, splitter, is_grouped_data=True,
                         scale_function=stacked_scale_function, n_jobs=n_jobs,
                         scorer=scorer, vote_type=vote_type)
        targ, pred = node.fit_transform(l0_predictions, l0_y, group=group,
                                        pred_type=l1_pred_type, group_pred=False)

        l1_group.append(node.group)
        l1_y.append(targ)
        l1_predictions.append(pred)

        l1_scores[key1]['standard'] = node.scores
        l1_scores[key1]['majority_vote'] = node.scores_group

    l1_predictions = np.concatenate(l1_predictions, axis=0)
    l1_y = np.concatenate(l1_y, axis=0)
    l1_group = np.concatenate(l1_group, axis=0)

    node = StackNode(stacked_estimator, splitter, is_grouped_data=True,
                    scale_function=stacked_scale_function, n_jobs=n_jobs,
                    scorer=scorer, vote_type=vote_type)

    node.fit(l1_predictions, l1_y, group=l1_group)

    return l0_scores, l1_scores, node.scores_group



