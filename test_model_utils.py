'''
Run unittests for doordash_utils
'''

import unittest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from pandas.util.testing import assert_series_equal
from sklearn import preprocessing
from model_utils import read_input
from model_utils import impute_data
from model_utils import convert_timestamp_to_seconds
from model_utils import one_hot_encoding
from model_utils import hash_encoding
from model_utils import split_train_test_data
from model_utils import scaling_type
from model_utils import split_train_test_validation_data
from model_utils import data_preprocessor

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.input_data, self.config_map = read_input(
            'Input/historical_data.csv', 'Input/data_config.json')
        self.nan_data = self.input_data.head(100)
        self.input_data = self.input_data.dropna()
        self.input_data = self.input_data.reset_index(drop=True)
        self.data = self.input_data.head(5)

    def test_read_input(self):
        self.assertTrue(len(self.input_data.index)
                        or len(self.config_map) != 0)

    def test_impute_data(self):
        actual_data = impute_data(self.nan_data, self.config_map)
        expected_data = pd.DataFrame(
            data={
                'store_primary_category': [
                    'american',
                    'mexican',
                    'american',
                    'american',
                    'american']})
        actual_data = actual_data['store_primary_category'].head(5).to_frame()
        assert_frame_equal(actual_data, expected_data)

    def test_convert_timestamp_to_seconds(self):
        actual_data = convert_timestamp_to_seconds(
            self.data, self.config_map['timestamp_columns'])
        expected_data = pd.DataFrame(
            data={
                'created_at': [
                    1423261457,
                    1423604965,
                    1424045495,
                    1423712206,
                    1422324756],
                'actual_delivery_time': [
                    1423265236,
                    1423608989,
                    1424047081,
                    1423714479,
                    1422327744]})
        assert_frame_equal(
            actual_data[self.config_map['timestamp_columns']], expected_data)

    def test_one_hot_encoding(self):
        actual_data = one_hot_encoding(
            self.data, self.config_map['encode_columns'])
        expected_data = pd.DataFrame(
            data={
                'market_id_1.0': [
                    1, 0, 0, 1, 1], 'order_protocol_1.0': [
                    1, 0, 0, 1, 1], 'order_protocol_2.0': [
                    0, 1, 0, 0, 0]})
        assert_frame_equal(
            actual_data[expected_data.columns].astype(np.int64), expected_data)

    def test_hash_enconding(self):
        actual_data = hash_encoding(
            self.data, self.config_map["hash_columns"], 5, 'string')
        expected_data = pd.DataFrame(data={'store_id_0': [7, 10, 10, 9, 9],
                                           'store_id_1': [9, 0, 0, 4, 4],
                                           'store_id_2': [-5, -2, -2, -7, -7],
                                           'store_id_3': [-2, 3, 3, 3, 3],
                                           'store_primary_category_0': [3, 1, 1, 2, 2],
                                           'store_primary_category_1': [0, 0, -2, -2, -2],
                                           'store_primary_category_2': [-1, -1, 0, 1, 1],
                                           'store_primary_category_3': [0, 1, 0, 0, 0]
                                           })
        assert_frame_equal(
            actual_data[expected_data.columns].astype(np.int64), expected_data)

    def test_read_input_none(self):
        self.assertRaises(ValueError, read_input)

    def test_impute_data_none(self):
        self.assertRaises(ValueError, impute_data)

    def test_split_train_test_validation_data(self):
        self.assertRaises(ValueError, split_train_test_validation_data)

    def test_split_train_test_data(self):
        self.assertRaises(ValueError, split_train_test_data)

    def test_data_preprocessor(self):
        self.assertRaises(ValueError, data_preprocessor)

    def test_scaling_type(self):
        scalar = scaling_type()
        self.assertIsInstance(scalar, preprocessing.StandardScaler)
