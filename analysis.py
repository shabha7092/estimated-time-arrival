""" Module for the Analysis """
__author__ = 'shabha'


import os
import argparse
import datetime as dt

from model_analysis import run_estimator_selection
from model_analysis import run_train_test_model
from model_analysis import run_train_validation_test_model
from model_analysis import run_fractional_stratification_model
from model_analysis import run_model_tuning
from model_analysis import run_prediction
from model import residual_plot
from model_utils import delete_output
from model_utils import write_data
from sklearn import linear_model



def main(config_path, data_to_predict_path, data_path):
    output_path = os.path.join(os.getcwd(), "Output/{}".format(dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    os.makedirs(output_path)
    estimator = linear_model.LinearRegression(n_jobs=-1)
    # print('Executing Estimator Selection')
    # estimator = run_estimator_selection(
    #        data_path, config_path)
    print('Executing Train / Test Model')
    model_properties = run_train_test_model(
            estimator, data_path, config_path, 20)
    print('Executing Train / Validation / Test Model')
    model_properties = run_train_validation_test_model(
            estimator, data_path, config_path, 20)
    # print('Executing Fractional Stratification Model')
    # model_properties = run_fractional_stratification_model(
    #        estimator, data_path, config_path, 5)
    print('Executing Model Tuning')
    tuned_properties = run_model_tuning(model_properties)
    print('Residual Plot')
    residual_plot(tuned_properties, output_path)
    print('Executing Prediction')
    predictions = run_prediction(data_to_predict_path, tuned_properties)
    print('Writing Predictions to File')
    write_data(predictions, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config',
            default=os.path.join(os.getcwd(), "Input/data_config.json"),
            help="path to data_config.json")
    parser.add_argument('--data_to_predict',
            default=os.path.join(os.getcwd(), "Input/data_to_predict.json"),
            help="path to data_to_predict.json")
    parser.add_argument('--historical_data',
            default=os.path.join(os.getcwd(), "Input/historical_data.csv"),
            help="path to historical_data.csv")
    args = parser.parse_args()
    main(args.data_config, args.data_to_predict, args.historical_data)
