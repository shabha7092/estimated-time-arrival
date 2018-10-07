'''
Run unittests for doordash_model_analysis
'''

import unittest
import pandas as pd
from sklearn import linear_model
from model_analysis import run_train_test_model
from model_analysis import run_train_validation_test_model
from model_analysis import run_fractional_stratification_model
from model_analysis import run_model_tuning
from model_analysis import run_estimator_selection
from model_analysis import run_prediction
from pandas.util.testing import assert_frame_equal


class TestModelAnalysis(unittest.TestCase):
    '''
    Class to test Model Analysis
    '''

    def setUp(self):
        self.data_path = 'Input/test_data.csv'
        self.prediction_data_path = 'Input/data_to_predict.json'
        self.config_path = 'Input/data_config.json'
        self.estimator = linear_model.LinearRegression(n_jobs=-1)

#     def test_estimator_selection(self):
#         estimator = run_estimator_selection(self.data_path, self.config_path)
#         self.assertIsInstance(estimator , linear_model.LinearRegression)

    def test_train_test_model(self):
        model_properties = run_train_test_model(
                self.estimator, self.data_path, self.config_path, seed=21)
        self.assertAlmostEqual(
                model_properties['training_rmse'],
                0.052222374677469124)
        self.assertAlmostEqual(
                model_properties['testing_rmse'],
                8.363558733860678)
        return


    def test_train_validation_test_model(self):
        model_properties = run_train_validation_test_model(
                self.estimator, self.data_path, self.config_path, seed=15)
        self.assertAlmostEqual(
                model_properties['training_rmse'],
                0.053897403934431924)
        self.assertAlmostEqual(
                model_properties['validation_rmse'],
                10.488012580363696)
        self.assertAlmostEqual(
                model_properties['testing_rmse'],
                0.05191548862703908)
        return

    def test_fractional_stratification_model(self):
        model_properties = run_fractional_stratification_model(
                self.estimator, self.data_path, self.config_path, seed=64)
        self.assertAlmostEqual(
                round(
                    model_properties['training_rmse'],
                    3),
                0.053)
        self.assertAlmostEqual(
                round(
                    model_properties['validation_rmse'],
                    3),
                0.053)
        self.assertAlmostEqual(
                round(
                    model_properties['testing_rmse'],
                    3),
                0.055)
        return

    def test_model_tuning(self):
        model_properties = run_train_validation_test_model(
                self.estimator, self.data_path, self.config_path, seed=15)
        tuned_properties = run_model_tuning(model_properties)
        self.assertAlmostEqual(
                tuned_properties['training_rmse'],
                0.054569030540759594)
        self.assertAlmostEqual(
                tuned_properties['validation_rmse'],
                10.487917475775953)
        self.assertAlmostEqual(
                tuned_properties['testing_rmse'],
                0.05252010262323503)
        return

    def test_prediction(self):
        model_properties = run_train_validation_test_model(
                self.estimator, self.data_path, self.config_path, seed=15)
        tuned_properties = run_model_tuning(model_properties)
        predictions = run_prediction(self.prediction_data_path, tuned_properties)  
        expected_data = pd.DataFrame(
            data={
                'delivery_id': [
                    '046433f6f07a2743d6c990f4ef61eaad',
                    '0ffb496c3eeb5eacec516a3cbcc9e569',
                    '3b04e68c88349a776013fcae3bc5aea6',
                    '345bb0c7adc228a776ae0a02a1b20a04',
                    'cffcd18445407b8027ed4484ff46726c'],
                'predicted_delivery_seconds': [
                    3433.443234,
                    3120.412867,
                    2759.408363,
                    2927.262850,
                    3164.216658]})
        assert_frame_equal(predictions.head(5), expected_data)
        return

    def test_train_test_model_invalid(self):
        model_properties = run_train_test_model(
                self.estimator, self.data_path, self.config_path, seed=73)
        self.assertIsNone(model_properties)
        return

    def test_train_validation_test_model_invalid(self):
        model_properties = run_train_validation_test_model(
                self.estimator, self.data_path, self.config_path, seed=16)
        self.assertIsNone(model_properties)

    def test_fractional_stratification_model_invalid(self):
        model_properties = run_fractional_stratification_model(
                self.estimator, self.data_path, self.config_path, seed=10)
        self.assertIsNone(model_properties)

    def test_estimator_selection_none(self):
        self.assertRaises(ValueError, run_estimator_selection)

    def test_train_test_model_none(self):
        self.assertRaises(ValueError, run_train_test_model)

    def test_train_validation_test_model_none(self):
        self.assertRaises(ValueError, run_train_validation_test_model)

    def test_fractional_stratification_model_none(self):
        self.assertRaises(ValueError, run_fractional_stratification_model)

    def test_model_tuning_none(self):
        self.assertRaises(ValueError, run_model_tuning)

    def test_prediction_none(self):
        self.assertRaises(ValueError, run_prediction)
