#!/usr/bin/env python

""" Module for the model """
__author__ = 'shabha'

import os
import matplotlib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from yellowbrick.regressor import ResidualsPlot
from doordash_utils import scaling_type

matplotlib.interactive(True)

def training(estimator=None, X_train=None, y_train=None, config_map=None):
    '''
    Method to train the estimator
    '''
    if estimator is None or X_train is None or y_train is None or config_map is None:
        raise ValueError(
            'Need Estimator, X_train, y_train and Config map as arguments!')
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_scaler = scaling_type(
        'min-max').fit(X_train[config_map['scale_columns']])
    y_scaler = scaling_type('min-max').fit(y_train[config_map['label']])
    X_train[config_map['scale_columns']] = X_scaler.transform(
        X_train[config_map['scale_columns']])
    y_train[config_map['label']] = y_scaler.transform(
        y_train[config_map['label']])
    model = clone(estimator)
    model.fit(X_train, y_train)
    rmse = calculate_error(X_train, y_train, model)
    return X_scaler, y_scaler, model, rmse


def calculate_rmse(
        X=None,
        y=None,
        X_scaler=None,
        y_scaler=None,
        model=None,
        config_map=None):
    '''
    Method to validate the model
    '''
    if X is None or y is None or X_scaler is None or y_scaler is None or model is None or config_map is None:
        raise ValueError(
            'Need X, Y, X_scaler, y_scaler, model and Config map as arguments !')
    X = X.copy()
    y = y.copy()
    X[config_map['scale_columns']] = X_scaler.transform(
        X[config_map['scale_columns']])
    y[config_map['label']] = y_scaler.transform(y[config_map['label']])
    rmse = calculate_error(X, y, model)
    return rmse


def calculate_error(X=None, y=None, model=None):
    '''
    returns residual and rmse error for already fit model
    '''
    if X is None or y is None or model is None:
        raise ValueError('Need X, Y, Model as arguments !')
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    return rmse


def residual_plot(model_properties=None, output_path=None):
    '''
    Method that shows the residual plot of the trained model
    '''
    if model_properties is None or output_path is None:
        raise ValueError('Need Model properties and Output path as arguments !')
    estimator = model_properties['estimator']
    X_train = model_properties['X_train']
    y_train = model_properties['y_train']
    X_validation = model_properties['X_validation']
    y_validation = model_properties['y_validation']
    config_map = model_properties['config_map']
    X_scaler = model_properties['X_scaler']
    y_scaler = model_properties['y_scaler']
    X_train[config_map['scale_columns']] = X_scaler.transform(
        X_train[config_map['scale_columns']])
    y_train[config_map['label']] = y_scaler.transform(
        y_train[config_map['label']])
    X_validation[config_map['scale_columns']] = X_scaler.transform(
        X_validation[config_map['scale_columns']])
    y_validation[config_map['label']] = y_scaler.transform(
        y_validation[config_map['label']])
    visualizer = ResidualsPlot(estimator)
    visualizer.fit(X_train.values, y_train.values)
    visualizer.score(X_validation.values, y_validation.values)
    visualizer.poof(outpath=os.path.join(output_path, 'residual_plot.png'))
    return None


def prediction(data=None, model_properties=None):
    '''
    Method for prediction 
    '''
    if data is None or model_properties is None:
         raise ValueError('Need Data to predict and Model Properties as arguments !')
    data = data.copy()
    config_map = model_properties['config_map']
    data[config_map['scale_columns']] = model_properties['X_scaler'].transform(data[config_map['scale_columns']])
    predictions = model_properties['model'].predict(data)
    predictions = model_properties['y_scaler'].inverse_transform(predictions)	
    return predictions
