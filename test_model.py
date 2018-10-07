'''
Run unittest for doordash_model
'''

import unittest
from sklearn import preprocessing
from sklearn import linear_model
from model_utils import read_input
from model_utils import data_preprocessor
from model_utils import split_train_test_validation_data
from model import training
from model import calculate_error
from model import calculate_rmse
from model import prediction

class TestModel(unittest.TestCase):
    ''' Test model class
    '''

    def setUp(self):
        self.data, self.config_map = read_input(
            'Input/test_data.csv', 'Input/data_config.json')
        self.estimator = linear_model.LinearRegression()
        self.data = data_preprocessor(self.data, self.config_map, 5, 'string')
        self.X_train, self.y_train, self.X_validation, self.y_validation, self.X_test, self.y_test = split_train_test_validation_data(
            self.data, self.config_map, 0.25, 0.2, 5)
        self.X_scaler, self.y_scaler, self.model, self.training_rmse = training(
            self.estimator, self.X_train, self.y_train, self.config_map)

    def test_training(self):
        self.assertIsInstance(self.X_scaler, preprocessing.MinMaxScaler)
        self.assertIsInstance(self.y_scaler, preprocessing.MinMaxScaler)
        self.assertIsInstance(self.model, linear_model.LinearRegression)

    def test_calculate_error(self):
        self.assertAlmostEqual(self.training_rmse, 0.01288907215822764)

    def test_validation_error(self):
        validation_rmse = calculate_rmse(
            self.X_validation,
            self.y_validation,
            self.X_scaler,
            self.y_scaler,
            self.model,
            self.config_map)
        self.assertAlmostEqual(validation_rmse, 0.0007615906548161986)

    def test_testing_error(self):
        testing_rmse = calculate_rmse(
            self.X_test,
            self.y_test,
            self.X_scaler,
            self.y_scaler,
            self.model,
            self.config_map)
        self.assertAlmostEqual(testing_rmse, 0.0007581966925813912)

    def test_training_none(self):
        self.assertRaises(ValueError, training)

    def test_prediction(self):
        self.assertRaises(ValueError, prediction)

    def test_calculate_error_none(self):
        self.assertRaises(ValueError, calculate_error)
