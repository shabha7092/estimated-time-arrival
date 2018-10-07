'''
Run test for fractional_stratification
'''

import unittest
import json as js
import pandas as pd
from fractional_stratification import create_omega_partitions
from fractional_stratification import update_partitions
from fractional_stratification import update_propotions
from fractional_stratification import fractional_stratification


class TestFractionalStratification(unittest.TestCase):

    def setUp(self):
        with open('Input/data_config.json') as config_file:
            config = js.load(config_file)
        self.config_map = dict(config)
        self.precision = 4
        self.S = [0.6, 0.2, 0.2]
        self.seed = 5
        self.omegas = []
        self.partitions = []
        self.selected_data = []
        self.rejected_data = []
        self.data = pd.DataFrame(
            data={
                'order_time': [
                    0.6,
                    0.5,
                    0.2,
                    0.8,
                    0.1,
                    0.3,
                    0.4,
                    0.15,
                    0.25,
                    0.35,
                    0.95,
                    0.45],
                'order_test': [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6]})
        self.omegas = create_omega_partitions(
            self.data, self.config_map, self.precision)
        self.selected_data, self.rejected_data, self.actual_partitions = update_partitions(
            self.omegas, self.S, self.partitions, self.selected_data, self.rejected_data, self.seed)

    def test_create_omega_partitions(self):
        expected_omegas = [
            [
                (0.1, 5.0), (0.15, 2.0), (0.2, 3.0)], [
                (0.25, 3.0), (0.3, 6.0), (0.35, 4.0)], [
                (0.4, 1.0), (0.45, 6.0), (0.5, 2.0)], [
                    (0.6, 1.0), (0.8, 4.0), (0.95, 5.0)]]
        self.assertListEqual(self.omegas, expected_omegas)

    def test_update_partitions(self):
        expected_partitions = [
            [(0.2, 3.0), (0.35, 4.0), (0.5, 2.0), (0.95, 5.0)], [], []]
        self.assertListEqual(self.actual_partitions, expected_partitions)

    def test_update_propotions(self):
        actual_splits = update_propotions(
            self.data,
            self.S,
            self.selected_data,
            self.rejected_data,
            self.actual_partitions)
        expected_splits = [
            0.3999999999999999,
            0.30000000000000004,
            0.30000000000000004]
        self.assertListEqual(actual_splits, expected_splits)

    def test_fractional_stratification(self):
        training_data, validation_data, testing_data = fractional_stratification(
            self.data, self.data.columns, self.precision, self.S, self.config_map)
        self.assertTrue(len(training_data.index) or len(
            validation_data.index) or len(validation_data.index) != 0)

    def test_fractional_stratification_none(self):
        self.assertRaises(ValueError, fractional_stratification)
