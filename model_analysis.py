# coding: utf-8

""" Module for the model analysis """
__author__ = 'shabha'


import sys
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from fractional_stratification import fractional_stratification
from model_utils import split_data
from model_utils import read_input
from model_utils import read_json_data
from model_utils import scaling_type
from model_utils import data_preprocessor
from model_utils import split_train_test_data
from model_utils import split_train_test_validation_data
from model_utils import impute_data
from model_utils import one_hot_encoding
from model_utils import hash_encoding
from model import training
from model import prediction
from model import calculate_rmse


def run_estimator_selection(
        data_path=None,
        config_path=None,
        Kfolds=10,
        n_jobs=-1):
    '''
    Execute's Cross Validator Model Analysis
    '''
    if data_path is None or config_path is None:
        raise ValueError('Need Data path and Config path as arguments !')
    estimators = []
    lr_class = LinearRegression(n_jobs=n_jobs)
    rf_class = RandomForestRegressor(n_jobs=n_jobs)
    nn_class = MLPRegressor()
    dt_class = DecisionTreeRegressor()
    estimators.append(lr_class)
    estimators.append(nn_class)
    estimators.append(dt_class)
    estimators.append(rf_class)
    return estimator_selection(estimators, data_path, config_path)


def estimator_selection(
        estimators=None,
        data_path=None,
        config_path=None,
        Kfolds=10,
        n_jobs=-1):
    '''
    Cross Validator Best Estimator Analysis
    '''
    if data_path is None or config_path is None or estimators is None:
        raise ValueError('Need Estimators, Data path and Config Path as arguments  !')
    data, config_map = read_input(data_path, config_path)
    data = data_preprocessor(data, config_map, 5, 'string')
    y = data[config_map['label']]
    data.drop(y[config_map['label']], axis=1, inplace=True)
    data[config_map["scale_columns"]] = scaling_type(
            'min-max').fit_transform(data[config_map["scale_columns"]])
    y[config_map["label"]] = scaling_type(
            'min-max').fit_transform(y[config_map["label"]])
    rmse_errors = []
    for estimator in estimators:
        mse = np.sum(-cross_val_score(estimator, data, y,
            scoring='neg_mean_squared_error', cv=Kfolds, n_jobs=n_jobs))
        rmse = np.sqrt(mse)
        rmse_errors.append(rmse)
    index_of_lowest_error = rmse_errors.index(min(rmse_errors))
    return estimators[index_of_lowest_error]


def run_train_test_model(
        estimator=None,
        data_path=None,
        config_path=None,
        num_iter=1,
        seed=None):
    '''
    Train/test Model Analysis
    '''
    if estimator is None or data_path is None or config_path is None:
        raise ValueError('Need Estimator, Data path and Config Path as arguments !')
    data, config_map = read_input(data_path, config_path)
    data = data_preprocessor(data, config_map, 5, 'string')
    training_map = {}
    for _ in range(0, num_iter):
        X_train, y_train, X_test, y_test = split_train_test_data(
                data, config_map, 0.3, seed)
        X_scaler, y_scaler, model, training_rmse = training(
                estimator, X_train, y_train, config_map)
        testing_rmse = calculate_rmse(
                X_test, y_test, X_scaler, y_scaler, model, config_map)
        if training_rmse < testing_rmse:
            model_properties = {}
            model_properties['estimator'] = estimator
            model_properties['config_map'] = config_map
            model_properties['X_train'] = X_train
            model_properties['y_train'] = y_train
            model_properties['X_test'] = X_test
            model_properties['y_test'] = y_test
            model_properties['X_scaler'] = X_scaler
            model_properties['y_scaler'] = y_scaler
            model_properties['model'] = model
            model_properties['training_rmse'] = training_rmse
            model_properties['testing_rmse'] = testing_rmse
            training_map[testing_rmse] = model_properties
    if(len(training_map) > 0):
        best_model_properties = training_map[min(training_map)]
        print('Best Model train error: {} | Best Model test error: {}'.format(
            round(best_model_properties['training_rmse'], 7),
            round(best_model_properties['testing_rmse'], 7)))
        return best_model_properties
    return None


def run_train_validation_test_model(
        estimator=None,
        data_path=None,
        config_path=None,
        num_iter=1,
        seed=None):
    '''
    Train/validation/test Model Analysis
    '''
    if estimator is None or data_path is None or config_path is None:
        raise ValueError('Need Estimator, Data path and Config Path as arguments !')
    data, config_map = read_input(data_path, config_path)
    data = data_preprocessor(data, config_map, 5, 'string')
    training_map = {}
    for _ in range(0, num_iter):
        X_train, y_train, X_validation, y_validation, X_test, y_test = split_train_test_validation_data(
                data, config_map, 0.25, 0.2, seed)
        X_scaler, y_scaler, model, training_rmse = training(
                estimator, X_train, y_train, config_map)
        validation_rmse = calculate_rmse(
                X_validation,
                y_validation,
                X_scaler,
                y_scaler,
                model,
                config_map)
        testing_rmse = calculate_rmse(
                X_test, y_test, X_scaler, y_scaler, model, config_map)
        if training_rmse < validation_rmse:
            model_properties = {}
            model_properties['estimator'] = estimator
            model_properties['config_map'] = config_map
            model_properties['X_train'] = X_train
            model_properties['y_train'] = y_train
            model_properties['X_validation'] = X_validation
            model_properties['y_validation'] = y_validation
            model_properties['X_test'] = X_test
            model_properties['y_test'] = y_test
            model_properties['X_scaler'] = X_scaler
            model_properties['y_scaler'] = y_scaler
            model_properties['model'] = model
            model_properties['training_rmse'] = training_rmse
            model_properties['validation_rmse'] = validation_rmse
            model_properties['testing_rmse'] = testing_rmse
            training_map[validation_rmse] = model_properties
    if len(training_map) > 0:
        best_model_properties = training_map[min(training_map)]
        print('Best Model train error: {} | Best Model validation error: {} | Best Model test error: {}'.format(
            round(best_model_properties['training_rmse'], 7),
            round(best_model_properties['validation_rmse'], 7),
            round(best_model_properties['testing_rmse'], 7)))
        return best_model_properties
    return None


def run_fractional_stratification_model(
        estimator=None,
        data_path=None,
        config_path=None,
        num_iter=1,
        seed=None):
    '''
    Fractional Stratification Model Analysis
    '''
    if estimator is None or data_path is None or config_path is None:
        raise ValueError('Need Estimator, Data path and Config Path as arguments !')
    data, config_map = read_input(data_path, config_path)
    data = data_preprocessor(data, config_map, 5, 'string')
    training_map = {}
    for _ in range(0, num_iter):
        training_data, validation_data, testing_data = fractional_stratification(
                data, data.columns, 4, [0.6, 0.2, 0.2], config_map, seed)
        X_train, y_train = split_data(training_data, config_map)
        X_validation, y_validation = split_data(validation_data, config_map)
        X_test, y_test = split_data(testing_data, config_map)
        X_scaler, y_scaler, model, training_rmse = training(
                estimator, X_train, y_train, config_map)
        validation_rmse = calculate_rmse(
                X_validation,
                y_validation,
                X_scaler,
                y_scaler,
                model,
                config_map)
        testing_rmse = calculate_rmse(
                X_test, y_test, X_scaler, y_scaler, model, config_map)
        if training_rmse < validation_rmse:
            model_properties = {}
            model_properties['estimator'] = estimator
            model_properties['config_map'] = config_map
            model_properties['X_train'] = X_train
            model_properties['y_train'] = y_train
            model_properties['X_validation'] = X_validation
            model_properties['y_validation'] = y_validation
            model_properties['X_test'] = X_test
            model_properties['y_test'] = y_test
            model_properties['X_scaler'] = X_scaler
            model_properties['y_scaler'] = y_scaler
            model_properties['model'] = model
            model_properties['training_rmse'] = training_rmse
            model_properties['validation_rmse'] = validation_rmse
            model_properties['testing_rmse'] = testing_rmse
            training_map[validation_rmse] = model_properties
    if(len(training_map) > 0):
        best_model_properties = training_map[min(training_map)]
        print('Best Model train error: {} | Best Model validation error: {} | Best Model test error: {}'.format(
            round(best_model_properties['training_rmse'], 7),
            round(best_model_properties['validation_rmse'], 7),
            round(best_model_properties['testing_rmse'], 7)))
        return best_model_properties
    return None


def run_model_tuning(model_properties=None):
    '''
    Tunes the Model based on Training and Validation error
    '''
    if model_properties is None:
        raise ValueError('Need Model Properties as argument !')
    alphas = np.logspace(-10, 1, 400)
    config_map = model_properties['config_map']
    X_train = model_properties['X_train']
    y_train = model_properties['y_train']
    X_validation = model_properties['X_validation']
    y_validation = model_properties['y_validation']
    X_test = model_properties['X_test']
    y_test = model_properties['y_test']
    tuning_map = {}
    for alpha in alphas:
        estimator = Ridge(alpha=alpha)
        X_scaler, y_scaler, model, training_rmse = training(
                estimator, X_train, y_train, config_map)
        validation_rmse = calculate_rmse(
                X_validation,
                y_validation,
                X_scaler,
                y_scaler,
                model,
                config_map)
        testing_rmse = calculate_rmse(
                X_test, y_test, X_scaler, y_scaler, model, config_map)
        tuning_properties = {}
        tuning_properties['estimator'] = estimator
        tuning_properties['config_map'] = config_map
        tuning_properties['X_scaler'] = X_scaler
        tuning_properties['y_scaler'] = y_scaler
        tuning_properties['model'] = model
        tuning_properties['training_rmse'] = training_rmse
        tuning_properties['validation_rmse'] = validation_rmse
        tuning_properties['testing_rmse'] = testing_rmse
        tuning_map[validation_rmse] = tuning_properties
    if(len(tuning_map) > 0):
        best_model_properties = tuning_map[min(tuning_map)]
        best_model_properties['config_map'] = config_map
        best_model_properties['X_train'] = X_train
        best_model_properties['y_train'] = y_train
        best_model_properties['X_validation'] = X_validation
        best_model_properties['y_validation'] = y_validation
        best_model_properties['X_test'] = X_test
        best_model_properties['y_test'] = y_test
        print('Best Model train error: {} | Best Model validation error: {} | Best Model test error: {}'.format(
            round(best_model_properties['training_rmse'], 7),
            round(best_model_properties['validation_rmse'], 7),
            round(best_model_properties['testing_rmse'], 7)))
        return best_model_properties
    return None


def run_prediction(
        data_path=None, 
        model_properties=None):
    '''
    Prediction's for the Model
    '''
    if data_path is None or model_properties is None:
        raise ValueError('Need Data path , Config pathand Model Properties as argument !')
    config_map = model_properties['config_map']
    data_to_predict = read_json_data(data_path)
    data = data_to_predict.copy()
    data = data.replace('NA', np.NaN)
    data[config_map['encode_columns']] = data[config_map['encode_columns']].astype(float)
    data[config_map['scale_columns']] = data[config_map['scale_columns']].astype(float)
    data = impute_data(data, config_map)
    data = one_hot_encoding(data, config_map['encode_columns'])
    data = hash_encoding(data, config_map['hash_columns'], 5, 'string')
    data = data[model_properties['X_train'].columns]
    predictions = prediction(data, model_properties)
    data_to_predict['predicted_delivery_seconds'] = list(predictions.flatten())
    predicted_data = data_to_predict[['delivery_id', 'predicted_delivery_seconds']]
    return predicted_data

