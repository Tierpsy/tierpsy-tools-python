#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 01 18:27:09 2021

@author: lferiani
"""
# %%
import warnings
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import factorial
from collections import defaultdict
from more_itertools import distinct_permutations, sort_together


def _count_shuffles(labels_series):
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


def _count_nested_shuffles(categories, labels):
    # make things into a dataframe for easier counting
    df = pd.DataFrame({'categories': categories, 'labels': labels})
    # in each category, we have n! permutations if all labels are different.
    # if some repeat, it's n! divided by the product of repetitions!
    # eg 11222 => 5!/(2!3!) = 10
    n_shuffles_each_category = []
    for _, gdf in df.groupby('categories'):
        n_shuffles_each_category.append(_count_shuffles(gdf['labels']))

    # print(n_shuffles_each_category)
    assert isinstance(n_shuffles_each_category[0], int)

    return n_shuffles_each_category


def check_asked_vs_possible_shuffles(categories, labels, n_asked_shuffles):
    """
    check_asked_vs_possible_shuffles Return True if the number of possible
    nested shuffles based on `categories`, `labels` is >= `n_asked_shuffles`

    Parameters
    ----------
    categories : iterable
        permutation blocks
    labels : iterable
        labels to shuffle within the categories
    n_asked_shuffles : scalar
        how many distinct shuffles we'd like to have

    Returns
    -------
    bool
        True if the number of shufflings asked is possible without repetition,
        False otherwise.
    """

    # get array of n_shuffles for each category
    n_shuffles_each_category = _count_nested_shuffles(categories, labels)

    # no entry of this should be 0. min should be 1
    assert all(ns > 0 for ns in n_shuffles_each_category)
    # first let's check if any of the categories alone yield more shuffles
    # than required
    if any(ns > n_asked_shuffles for ns in n_shuffles_each_category):
        return True
    # if that failed, let's gradually multiply (to avoid overflow) and check
    # each time

    n_nested_shuffles = 1
    for ss in n_shuffles_each_category:
        new_value = n_nested_shuffles * ss
        # first, let's see if this new value is larger than what we asked
        if new_value > n_asked_shuffles:
            return True
        # second, let's check we're not overflowing
        if new_value < n_nested_shuffles:
            print("Overflow counting shuffles ==> enough possible shuffles")
            return True
        # if no conditions are met, let's store new value and move on
        n_nested_shuffles = new_value

    # if we get to the end of the for loop without finding enough shuffles,
    return False


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
    seen = set()
    max_attempts = n_shuffles * 10
    for i in range(n_shuffles):
        # every time, make sure the permutation is new.
        # you have max_attampts for that to happen
        ac = 0
        while ac < max_attempts:
            shuffled_labs = np.concatenate([
                np.random.permutation(labs)
                for labs in labels_by_category.values()
            ], axis=0)
            if tuple(shuffled_labs) not in seen:
                seen.add(tuple(shuffled_labs))
                break
            ac += 1
        assert ac < max_attempts, (
            f"after {max_attempts} tries, cannot find an unseen shuffling. "
            f"Are there enough datapoints for {n_shuffles} distinct shuffles?"
            )
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
    # TODO: return warning if any category has 1 element and an error if all do
    # rearrange the data in a format that makes things easier, using a dict
    # this is equivalent to use a groupby but faster and not pandas
    labs_per_cat = defaultdict(list)
    for key, val in zip(categories, labels):
        labs_per_cat[key].append(val)
    # can we return as many shuffles as we were asked?
    if n_shuffles != 'all':
        is_enough_possible_shuffles = check_asked_vs_possible_shuffles(
            categories, labels, n_shuffles)
        if not is_enough_possible_shuffles:
            warnings.warn(
                f'Data do not support {n_shuffles} random shuffles, '
                f'returning all distinct shuffles instead')
            n_shuffles = 'all'

    if n_shuffles == 'all':
        nested_shufflings_iter = iterate_all_nested_shufflings(labs_per_cat)
    else:
        nested_shufflings_iter = iterate_n_nested_shufflings(
            labs_per_cat, n_shuffles=n_shuffles)

    return nested_shufflings_iter


# %%

if __name__ == '__main__':

    def check_endings(_cats, _labs):
        # check if shuffling has broken cat <==> label
        assert all(
            [cat == lab.split('_')[-1] for cat, lab in zip(_cats, _labs)])

    from pprint import pprint

    n_days = 3

    labels = ['N2', 'N2', 'mutant', 'mutant']
    labels = [f'{lab}_day{d}' for d in range(n_days, 0, -1) for lab in labels]
    categories = [lab.split('_')[-1] for lab in labels]
    check_endings(categories, labels)

    print("unshuffled: ")
    pprint(list(zip(categories, labels)))

    for sc, shuffled_labels in enumerate(
            iterate_nested_shufflings(categories, labels, n_shuffles='all')):

        check_endings(categories, shuffled_labels)

        print(f"shuffle {sc}: ")
        pprint(list(zip(categories, shuffled_labels)))

    # %%

    n_days = 3
    long_labels = ['N2', 'N2', 'mutant', 'mutant'] * 10
    long_labels = [
        f'{lab}_day{d}' for d in range(n_days, 0, -1) for lab in long_labels]

    long_categories = [lab.split('_')[-1] for lab in long_labels]
    check_endings(long_categories, long_labels)

    # expect (40! / (20! * 20!))^3 about 1E33 shufflings possible!
    # so only do 10k

    for sc, shuffled_long_labels in enumerate(
            iterate_nested_shufflings(
                long_categories, long_labels, n_shuffles=10000)):

        check_endings(long_categories, shuffled_long_labels)

# %%
    n_days = 3
    medium_labels = ['N2', 'N2', 'mutant', 'mutant'] * 2
    medium_labels = [
        f'{lab}_day{d}' for d in range(n_days, 0, -1) for lab in medium_labels]

    medium_categories = [lab.split('_')[-1] for lab in medium_labels]
    check_endings(medium_categories, medium_labels)

    # expect (8! / (4! * 4!))^3 => 343000 shufflings possible!
    # so only do 1e5
    n_shuffles = int(4e5)

    for sc, shuffled_medium_labels in enumerate(
            tqdm(
                iterate_nested_shufflings(
                    medium_categories, medium_labels, n_shuffles=n_shuffles),
                total=n_shuffles)
            ):

        check_endings(medium_categories, shuffled_medium_labels)





# %%
