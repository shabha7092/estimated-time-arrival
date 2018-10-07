#!/usr/bin/env python

""" Split the data on a value distribution """
__author__ = 'shabha'
__reference__ = 'http://scottclowe.com/2016-03-19-stratified-regression-partitions/'

import random
import bisect
import pandas as pd
import numpy as np
from math import floor


def fractional_stratification(
        data=None,
        columns=None,
        precision=None,
        S=None,
        config_map=None,
        seed=None):
    '''
    Split's the data based on distribution
    '''
    if data is None or columns is None or precision is None or S is None or config_map is None:
        raise ValueError(
            'Need Data, Columns, Precision, Split size, Config map as arguments !')
    return fractional_stratification_helper(
        data, columns, precision, S, config_map, [], seed)


def fractional_stratification_helper(
        data=None,
        columns=None,
        precision=None,
        S=None,
        config_map=None,
        partitions=None,
        seed=None):
    '''
    Helper function for stratification
    '''
    if precision <= 1 or len(data.index) == 0:
        if len(data.index) != 0:
            weighted_random_partition(data, 0, S, config_map, partitions)
        training_data = pd.DataFrame(partitions[0], columns=columns)
        validation_data = pd.DataFrame(partitions[1], columns=columns)
        testing_data = pd.DataFrame(partitions[2], columns=columns)
        return training_data, validation_data, testing_data
    omegas = create_omega_partitions(data, config_map, precision)
    selected_data, rejected_data, partitions = update_partitions(
        omegas, S, partitions, [], [], seed)
    S = update_propotions(data, S, selected_data, rejected_data, partitions)
    precision = int(floor(precision) / 2)
    data = pd.DataFrame(rejected_data, columns=columns)
    return fractional_stratification_helper(
        data, columns, precision, S, config_map, partitions)


def create_omega_partitions(data=None, config_map=None, precision=None):
    '''
    Creates the temporary partitions
    '''
    sorted_data = sort_data_frame(data, config_map)
    split = int(len(sorted_data.index) / precision)
    omegas = []

    for j in range(1, precision + 1):
        omega_j = [tuple(x) for x in sorted_data.head(split).values]
        omegas.append(omega_j)
        sorted_data.drop(sorted_data.index[:split], inplace=True)
    return omegas


def update_partitions(
        omegas=None,
        S=None,
        partitions=None,
        selected_data=None,
        rejected_data=None,
        seed=None):
    '''
    Updates the final output partitions
    '''
    for j, omega_j in enumerate(omegas):
        q_j = len(omega_j)
        for i, si in enumerate(S):
            assigned = floor(q_j * si)
            random.seed(seed)
            selected_i = random.sample(omega_j, assigned)
            rejected_j = [item for item in omega_j if item not in selected_i]
            omega_j = rejected_j
            if len(selected_data) <= i:
                selected_data.append([])
            selected_data[i] = selected_data[i] + selected_i
            if len(partitions) <= i:
                partitions.append([])
            partitions[i] = partitions[i] + selected_i

        rejected_data = rejected_data + omega_j
    return selected_data, rejected_data, partitions


def update_propotions(
        data=None,
        S=None,
        selected_data=None,
        rejected_data=None,
        partitions=None):
    '''
    Update propotions for the next iteration
    '''
    for k in range(0, len(S)):
        if len(rejected_data) > 0:
            S[k] = (S[k] * len(data) - len(selected_data[k])) / \
                len(rejected_data)
    return S


def weighted_random_partition(
        data=None,
        partition_index=None,
        S=None,
        config_map=None,
        partitions=None):
    '''
    Splits the remaining data in to final partitons using cumilative distribution function
    '''
    if(len(data.index) == 0):
        return partitions
    sorted_data = sort_data_frame(data, config_map)
    cum_weights = list(np.cumsum(S))
    random_number = np.random.random()
    index = bisect.bisect(cum_weights, random_number)
    if(index < len(sorted_data.index)):
        selected = [tuple(data.iloc[index])]
        partitions[partition_index %
                   len(S)] = partitions[partition_index %
                                        len(S)] + selected
        sorted_data = sorted_data.drop(index=index)
        sorted_data = sorted_data.reset_index(drop=True)
    return weighted_random_partition(
        sorted_data,
        partition_index + 1,
        S,
        config_map,
        partitions)


def sort_data_frame(data=None, config_map=None):
    '''
    Sorts the dataframe based on a column
    '''
    data = data.copy()
    sorted_data = data.sort_values(by=config_map['label'])
    sorted_data = sorted_data.reset_index(drop=True)
    return sorted_data
