#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 01 18:27:09 2021

@author: lferiani
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from math import factorial
from collections import defaultdict
from more_itertools import distinct_permutations, sort_together


def count_shuffles(labels_series):
    # this function looks weird, but it's written to keep things in integers
    # as integers in python are unbounded and should not have overflow issues
    if not isinstance(labels_series, pd.Series):
        labels_series = pd.Series(labels_series)
    # shuffles if all unique values
    n_shuffles = factorial(len(labels_series))
    # correct for repeated values. keeping it all in integers here
    n_repeated = 1
    for f in labels_series.value_counts().apply(factorial).values:
        n_repeated = n_repeated * int(f)
    n_shuffles = n_shuffles // n_repeated

    return n_shuffles


def count_nested_shuffles(categories, labels):
    # make things into a dataframe for easier counting
    df = pd.DataFrame({'categories': categories, 'labels': labels})
    # in each category, we have n! permutations if all labels are different.
    # if some repeat, it's n! divided by the product of repetitions!
    # eg 11222 => 5!/(2!3!) = 10
    n_shuffles_each_category = df.groupby(
        'categories')['labels'].apply(count_shuffles).values.astype(int)
    # print(n_shuffles_each_category)
    assert n_shuffles_each_category[0].dtype == int

    n_nested_shuffles = 1
    for ss in n_shuffles_each_category:
        n_nested_shuffles = n_nested_shuffles * ss

    return n_nested_shuffles


def is_non_decreasing(an_iterable):
    for i, el in enumerate(an_iterable[1:]):
        # i is 0 when el is an_iterable[1],
        # so i already points to the previous element
        if el < an_iterable[i]:
            return False
    return True


def is_non_increasing(an_iterable):
    for i, el in enumerate(an_iterable[1:]):
        # i is 0 when el is an_iterable[1],
        # so i already points to the previous element
        if el > an_iterable[i]:
            return False
    return True


def is_monotonic(an_iterable):
    """
    is_monotonic return True if `an_iterable` is either non-decreasing or
    non-increasing
    """
    return (is_non_increasing(an_iterable) or is_non_decreasing(an_iterable))


def iterate_all_nested_shufflings(labels_by_category):
    """
    iterate_all_nested_shufflings Returns all possible distinct shufflings
        (i.e. accounting for repeated values) respecting
        the structure in `labels_by_category`, which is a dictionary.
        This is a private function, not meant to be used directly.

    Parameters
    ----------
    labels_by_category : dictionary
        Dictionary where the keys are categories and the values are arrays
        of labels

    Returns
    -------
    nested_shufflings_iter : iterator
        Looping on this returns arrays of shuffled labels.
        Categories are not returned, but the labels
        are in the order the categories come in.
        So, the first n elements will
        be shuffled labels from the first category, then from the second one,
        etc.
    """
    # now create a dictionary of iterators
    # each iterator shuffles within the category
    iter_per_cat = dict()
    for cat, labs in labels_by_category.items():
        iter_per_cat[cat] = distinct_permutations(labs)
    # create a "big" iterator now (cross product of iterators)
    # This could be done without the previous dict but easier to read
    nested_shufflings_iter = itertools.product(*iter_per_cat.values())

    # nested_shufflings_iter would return a tuple of tuples.
    # we want an array instead.
    # That's achieved by np.ravel, so we need to map ravel to the iterator
    # The iterator will now yield flattened arrays
    nested_shufflings_iter = map(np.ravel, nested_shufflings_iter)

    return nested_shufflings_iter


def iterate_n_nested_shufflings(labels_by_category, n_shuffles):
    """
    iterate_n_nested_shufflings generator that returns n_shuffles respecting
        the structure in `labels_by_category`, which is a dictionary.
        This is a private function, not meant to be used directly.

    Parameters
    ----------
    labels_by_category : dictionary
        Dictionary where the keys are categories and the values are arrays
        of labels
    n_shuffles : a scalar
        how many within-category permutations to return

    Yields
    -------
    shuffled_labs : array
        Looping on this generator return arrays of shuffled labels.
        Categories are not returned, but the labels are in the order
        the categories come in.
        So, the first n elements will be shuffled labels from
        the first category, then from the second one, etc.
    """
    for i in range(n_shuffles):
        shuffled_labs = np.concatenate([
            np.random.permutation(labs) for labs in labels_by_category.values()
        ], axis=0)
        # shuffled_labs = np.ravel(shuffled_labs)
        yield shuffled_labs


def iterate_nested_shufflings(categories, labels, n_shuffles='all'):
    """
    iterate_all_nested_shufflings Create within-block permutations
        of `labels` (ignoring repeated values). Blocks are defined as having
        the same value in `categories`. Entire blocks are not shuffled.
        If there are n categories, and within each category N permutations are
        possible, there are at most N^n possible shufflings of labels.
        To keep this function speedy, categories needs to be sorted
        (technically, the algorithm would work ass long as blocks are
        contiguous, but sortedness is easier to test for)

    Parameters
    ----------
    categories : iterable, np.array is best
        descriptor of blocks of `labels`. `labels` will only be shuffled within
        blocks, not across blocks
    labels : iterable, np.array is best
        labels to shuffle within blocks defined by `categories`
    n_shuffles : scalar, or 'all'
        how many shufflings to return

    Returns
    -------
    iterator (a map or a generator)
        consuming the iterator yields arrays with all possible permutations
        respecting the block structure defined by categories:
        The first n elements will be shuffled labels from
        the first category, then from the second one, etc.
    """
    if not isinstance(categories, np.ndarray):
        categories = np.array(categories)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    assert is_monotonic(categories), 'sort by categories first!'
    # rearrange the data in a format that makes things easier, using a dict
    # this is equivalent to use a groupby but faster and not pandas
    labs_per_cat = defaultdict(list)
    for key, val in zip(categories, labels):
        labs_per_cat[key].append(val)
    # Removing following bit as it's giving more problems than it solves
    # # can we return as many shuffles as we were asked?
    # if n_shuffles != 'all':
    #     n_possible_shuffles = count_nested_shuffles(categories, labels)
    #     is_too_few = n_possible_shuffles <= n_shuffles
    #     if is_too_few:
    #         warnings.warn(
    #             f'Data do not support {n_shuffles} random shuffles, '
    #             f'returning {n_possible_shuffles} distinct ones instead')
    #         n_shuffles = 'all'

    if n_shuffles == 'all':
        nested_shufflings_iter = iterate_all_nested_shufflings(labs_per_cat)
    else:
        nested_shufflings_iter = iterate_n_nested_shufflings(
            labs_per_cat, n_shuffles=n_shuffles)

    return nested_shufflings_iter
