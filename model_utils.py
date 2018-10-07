#!/usr/bin/env python

""" Utility Module of helper functions"""
__author__ = 'shabha'

import os
import datetime as dt
import json as js
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split


def read_input(data_path=None, config_path=None):
    '''
    return's data and config
    '''
    if data_path is None or config_path is None:
        raise ValueError('Need Data path and Config path as arguments !')
    data = read_data(data_path)
    with open(config_path) as config_file:
        config = js.load(config_file)
    config_map = dict(config)
    return data, config_map


def read_data(data_path=None):
    '''
    return's dataframe from csvfile
    '''
    if data_path is None:
        raise ValueError('Need csv file path as argument !')
    data = pd.read_csv(data_path)
    data = data.copy()
    return data


def read_json_data(data_path=None):
    '''
    return's dataframe from csvfile
    '''
    if data_path is None:
        raise ValueError('Need json file path as argument !')
    data = pd.read_json(data_path, precise_float=True, lines=True)
    data = data.copy()
    return data

def impute_data(data=None, config_map=None, strategy='mean'):
    '''
    imputes the data and return's dataframe
    '''
    if data is None or config_map is None:
        raise ValueError('Need Data and Config map as arguments !')
    data = data.copy()
    if set(config_map['scale_columns']).issubset(data.columns):
        if strategy == 'mean':
            data[config_map['scale_columns']] = data[config_map['scale_columns']].fillna(
                data[config_map['scale_columns']].mean())
        elif strategy == 'median':
            data[config_map['scale_columns']] = data[config_map['scale_columns']].fillna(
                data[config_map['scale_columns']].median())
        elif strategy == 'most_frequent':
            for column in config_map['scale_columns']:
                data[column] = data[column].fillna(
                    data[column].value_counts().idxmax())
    if set(config_map['categorical_columns']).issubset(data.columns):
        for column in config_map['categorical_columns']:
            data[column] = data[column].fillna(
                data[column].value_counts().idxmax())
    return data


def convert_timestamp_to_seconds(data=None, columns=None):
    '''
    convert timestamps in dataframe to unix seconds
    '''
    if data is None or columns is None:
        raise ValueError('Need Data and Columns as arguments !')
    data = data.copy()
    data[columns] = data[columns].applymap(lambda x: int(
        (pd.to_datetime(x) - dt.datetime(1970, 1, 1)).total_seconds()))
    return data


def one_hot_encoding(data=None, columns=None):
    '''
    one hot encoding for categorical data
    '''
    if data is None or columns is None:
        raise ValueError('Need Data and Columns as arguments !')
    data = data.copy()
    data = pd.concat([data, pd.get_dummies(
        data=data[columns], columns=columns)], axis=1)
    data.drop(data[columns], axis=1, inplace=True)
    # avoid 'no multicollinearity'
    for i in range(0, len(columns)):
        encoded_len = np.count_nonzero(data.columns.str.contains(columns[i]))
        label = columns[i] + "_" + str(float(encoded_len))
        data.drop(data[[label]], axis=1, inplace=True)
    return data


def hash_encoding(data=None, columns=None, n_features=None, input_type=None):
    '''
     Hash Encoding for categorical data
    '''
    if data is None or columns is None or n_features is None or input_type is None:
        raise ValueError(
            'Need Data, Columns, n_features and  input_type as arguments !')
    data = data.copy()
    hasher = FeatureHasher(n_features, input_type)
    labels = [
        columns[i] +
        '_' +
        str(j) for i in range(
            0,
            len(columns)) for j in range(
            0,
            n_features)]
    hashed_list = list(
        map(lambda column: hasher.fit_transform(data[column]).toarray(), columns))
    data.reset_index(drop=True, inplace=True)
    data = pd.concat([data, pd.DataFrame(np.concatenate(
        hashed_list, axis=1), columns=labels)], axis=1)
    data.drop(data[columns], axis=1, inplace=True)
    # avoid 'no multicollinearity'
    for i in range(0, len(columns)):
        encoded_len = np.count_nonzero(data.columns.str.contains(columns[i]))
        label = columns[i] + "_" + str(encoded_len - 1)
        data.drop(data[[label]], axis=1, inplace=True)
    return data


def split_train_test_validation_data(
        data=None,
        config_map=None,
        validation_size=None,
        test_size=None,
        random_state=None):
    '''
    spilts data between training and validation and testing
    '''
    if data is None or config_map is None or validation_size is None or test_size is None:
        raise ValueError(
            'Need Data, Config map, Validation size, Test Size as arguments !')
    data = data.copy()
    y = data[config_map['label']]
    X = data.drop(config_map['label'], axis=1)
    X, X_test, y, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=validation_size, random_state=random_state)
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def split_train_test_data(
        data=None,
        config_map=None,
        test_size=None,
        random_state=None):
    '''
    spilts data between training and validation and testing
    '''
    if data is None or map is None or test_size is None:
        raise ValueError('Need Data, Config map, Test size as arguments !')
    data = data.copy()
    y = data[config_map['label']]
    X = data.drop(config_map['label'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, y_train, X_test, y_test


def data_preprocessor(
        data=None,
        config_map=None,
        n_features=None,
        input_type=None):
    '''
     preproccess the data for model analysis
    '''
    if data is None or config_map is None or n_features is None or input_type is None:
        raise ValueError(
            'Need Data, Config map, n_features and  input_type as arguments !')
    data = data.copy()
    data = impute_data(data, config_map)
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = convert_timestamp_to_seconds(data, config_map['timestamp_columns'])
    data = pd.concat([data, pd.DataFrame(data[config_map['timestamp_columns'][1]] -
                                         data[config_map['timestamp_columns'][0]], columns=config_map['label'])], axis=1)
    data.drop(data[config_map['timestamp_columns']], axis=1, inplace=True)
    data = one_hot_encoding(data, config_map['encode_columns'])
    data = hash_encoding(
        data,
        config_map['hash_columns'],
        n_features,
        input_type)
    return data


def scaling_type(type=None):
    '''
    selects scaling method based on the type
    '''
    if type == 'min-max':
        return preprocessing.MinMaxScaler()
    elif type == 'robust':
        return preprocessing.RobustScaler()
    elif type == 'normalizer':
        return preprocessing.Normalizer()
    else:
        return preprocessing.StandardScaler()


def split_data(data=None, config_map=None):
    '''
    splits the data into X and y dataframes
    '''
    if data is None or config_map is None:
        raise ValueError('Need Data and Config map as arguments !')
    data = data.copy()
    y = data[config_map["label"]]
    X = data.drop(config_map["label"], axis=1)
    return X, y


def write_data(data=None, output_path=None, sep=','):
    '''
    Method to write the data to file
    '''
    if data is None or output_path is None:
        raise ValueError('Need Data and Output Path as arguments !')
    data.to_csv(os.path.join(output_path, 'predictions.tsv'), sep = sep)
    return None
