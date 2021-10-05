"""
Testing code for Churn model
Author: Arturo Polanco
Date: October 2021
"""

import logging
import churn_library


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("data/BankChurners.csv")
        logging.info("SUCCESS: Testing import_data.")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_eda: The file wasn't found")

    try:
        df = import_data("data/BankChurners.csv")
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("SUCCESS: Dimensions of dataframe are bigger than 0.")

    except AssertionError:
        logging.error(
            "ERROR: Testing import_data: The file doesn't appear to have rows and columns")


def test_eda(perform_eda=churn_library.perform_eda):
    '''
    test perform eda function
    '''

    eda_images = ['distplot_total_trans_ct.png',
                  'maritalStatus_count.png',
                  'customer_age.png',
                  'churn_hist.png',
                  'corr_heatmap.png']

    for image in eda_images:
        try:
            with open(f"images/eda/{image}", 'r'):
                logging.info(f"SUCCESS: EDA generated {image} image.")
        except FileNotFoundError:
            logging.error("ERROR: One or more images are missing from EDA.")


def test_encoder_helper(
        datafrane,
        encoder_helper=churn_library.encoder_helper):
    '''
    test encoder helper
    '''
    new_columns = ['Gender_Churn',
                   'Education_Level_Churn',
                   'Marital_Status_Churn',
                   'Income_Category_Churn',
                   'Card_Category_Churn']

    for column in new_columns:
        try:
            assert(column in datafrane.columns.tolist())
            logging.info(f"SUCCESS: encoder generated {column} column.")
        except AssertionError:
            logging.error(
                "ERROR: One or more columns are missing from encoding.")


def test_perform_feature_engineering(
        datafrane,
        perform_feature_engineering=churn_library.perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    for column in keep_cols:
        try:
            assert(column in datafrane.columns.tolist())
            logging.info(
                f"SUCCESS: feature engineering generated {column} column.")
        except AssertionError:
            logging.error(
                "ERROR: One or more columns are missing from feature engineering.")


def test_train_models(train_models=churn_library.train_models):
    '''
    test train_models
    '''
    list_models = ['logistic_model.pkl', 'rfc_model.pkl']

    for model in list_models:
        try:
            with open(f"models/{model}", 'r'):
                logging.info(f"SUCCESS: Train models generated {model} model.")
        except FileNotFoundError:
            logging.error(
                "ERROR: One or more models are missing from train models.")


if __name__ == "__main__":
    test_import(churn_library.import_data)
    churn_df = churn_library.import_data("data/BankChurners.csv")
    test_eda()
    trainX, testX, trainY, testY = churn_library.perform_feature_engineering(
        churn_df)
    test_encoder_helper(trainX)
    test_perform_feature_engineering(trainX)
    test_train_models()
