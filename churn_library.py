"""
This module predicts churn of credit card customers
Author: Arturo Polanco
Date: October 2021
"""

# import necessary libraries
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth_data):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    dataframe = pd.read_csv(pth_data)
    # create output Churn variable
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dataframe


def perform_eda(input_df, img_directory="images/eda"):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # exploratory data analysis
    print("Exploration head of the dataframe: \n", input_df.head())
    print("Shape of the dataframe: \n", input_df.shape)
    print("Number of Null values: \n", input_df.isnull().sum())
    print("Description dataframe: \n", input_df.describe())

    # create folder for storing EDA figures
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)

    # saving EDA figures
    # Saving churn histogram figure
    churn_hist_dir = os.path.join(img_directory, "churn_hist.png")
    input_df["Churn"].hist().figure.savefig(churn_hist_dir)
    plt.close()

    # Saving customer age histogram figure
    custm_age_hist_dir = os.path.join(img_directory, "customer_age.png")
    input_df['Customer_Age'].hist().figure.savefig(
        custm_age_hist_dir)
    plt.close()

    # Saving marital statues value count bar plot
    marital_stat_bar_dir = os.path.join(
        img_directory, "maritalStatus_count.png")
    input_df.Marital_Status.value_counts('normalize').plot(
        kind='bar').figure.savefig(marital_stat_bar_dir)
    plt.close()

    # Saving total transaction counts
    total_tran_displot_dir = os.path.join(
        img_directory, "distplot_total_trans_ct.png")
    sns.distplot(input_df['Total_Trans_Ct']).figure.savefig(
        total_tran_displot_dir)
    plt.close()

    # Saving correlation heatmap dataframe
    corr_heatmap_dir = os.path.join(img_directory, "corr_heatmap.png")
    sns.heatmap(input_df.corr(), annot=False, cmap='Dark2_r',
                linewidths=2).figure.savefig(corr_heatmap_dir)
    plt.close()


def encoder_helper(input_df, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # define response
    if response is None:
        response = "_Churn"

    # traverse list of categories
    for category in category_lst:
        new_column_name = category + response
        column_list = []
        grouped_values = input_df.groupby(category).mean()["Churn"]

        # traverse values in column
        for value in input_df[category]:
            column_list.append(grouped_values.loc[value])
        input_df[new_column_name] = column_list

    return input_df


def perform_feature_engineering(raw_df, response="train"):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # create extra features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    preprocesed_df = encoder_helper(raw_df, cat_columns)

    # separate predictors and target value
    # select the variables that will be used by the model
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
    input_features = pd.DataFrame()
    input_features[keep_cols] = preprocesed_df[keep_cols]
    # select churn column
    output_variable = preprocesed_df['Churn']

    # split the dataset for training and testing
    train_features, test_features, train_output, test_output = train_test_split(
        input_features, output_variable, test_size=0.3, random_state=42)

    return train_features, test_features, train_output, test_output


def train_models(
        train_features,
        test_features,
        train_output,
        test_output,
        models_directory="models"):
    '''
    train, store model results: images + scores, and store models
    input:
              train_features: X training data
              test_features: X testing data
              train_output: y training data
              test_output: y testing data
    output:
              rfc_model: Random Forest Classifier
              lr_model: Logistic Regression Classifier
              model_predictions: Predictions using previous models
    '''
    # instantiate the models
    random_forest_model = RandomForestClassifier(random_state=42)
    logistic_refression_model = LogisticRegression()

    # define parameters of gridsearch
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Training machine learning models
    # Find best parameters for random forest
    cv_rfc = GridSearchCV(
        estimator=random_forest_model,
        param_grid=param_grid,
        cv=5)
    # training random forest model
    cv_rfc.fit(train_features, train_output)
    # train logistic regression model
    logistic_refression_model.fit(train_features, train_output)

    # predict samples of training and testing sets
    model_predictions = generate_predictions(
        cv_rfc, logistic_refression_model, train_features, test_features)

    # create folder for storing ml models
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    # save best model random forest and logistic regression
    cv_rfc_dir = os.path.join(models_directory, "rfc_model.pkl")
    joblib.dump(cv_rfc.best_estimator_, cv_rfc_dir)
    log_reg_dir = os.path.join(models_directory, "logistic_model.pkl")
    joblib.dump(logistic_refression_model, log_reg_dir)

    # load models
    rfc_model = joblib.load(cv_rfc_dir)
    lr_model = joblib.load(log_reg_dir)

    return rfc_model, lr_model, model_predictions


def generate_predictions(rfc_model, lr_model, train_features, test_features):
    '''
    Use models to generate predictions using train and test features
    input:
              rfc_model: Random Forest Model
              lr_model: Logistic Regression Model
              train_features: Features training set
              test_features: Features test set
    output:
              train_preds_rf: Predictions training set using RFC
              test_preds_rf: Predictions test set using RFC
              train_preds_lr: Predictions training set using LR
              test_preds_lr: Predictions test set using LR
    '''
    # Predictions random forest classifier
    train_preds_rf = rfc_model.best_estimator_.predict(train_features)
    test_preds_rf = rfc_model.best_estimator_.predict(test_features)

    # Predictions logistic regression
    train_preds_lr = lr_model.predict(train_features)
    test_preds_lr = lr_model.predict(test_features)

    return train_preds_rf, test_preds_rf, train_preds_lr, test_preds_lr


def classification_report_image(y_train,
                                y_test,
                                X_test,
                                model_predictions,
                                rfc_model,
                                lr_model,
                                results_directory="images/results"):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            X_test: test features of train set
            model_predictions: Predictions using RF and LR
            rfc_model: Random morest model
            lr_model: Logistic regression model
            results_directory: Directory for storing result figures

    output:
             None
    '''
    # Unpack model predictions
    train_preds_rf, test_preds_rf, train_preds_lr, test_preds_lr = model_predictions

    # Print classification report random forest
    print('random forest results')
    print('test results')
    print(classification_report(y_test, test_preds_rf))
    print('train results')
    print(classification_report(y_train, train_preds_rf))

    # print classification report logistic regression
    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, test_preds_lr))
    print('train results')
    print(classification_report(y_train, train_preds_lr))

    # create folder for storing clasification report results
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # store figure images AUC curve
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plot_roc_curve(rfc_model, X_test, y_test, ax=lrc_plot.ax_, alpha=0.8)

    # create auc figure directory
    auc_plot_dir = os.path.join(results_directory, "auc_iamge.png")
    # save figure
    plt.savefig(auc_plot_dir)
    plt.close()


def feature_importance_plot(model, test_features, output_pth="images/results"):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            test_features: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # calculate the shapley values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_features)
    shap.summary_plot(shap_values, test_features, plot_type="bar", show=False)

    # create folder for storing shap values figure
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)

    # create directory of shap values figure and save
    shap_dir = os.path.join(output_pth, "shap_values.png")
    plt.savefig(shap_dir)
    plt.close()

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [test_features.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(test_features.shape[1]), importances[indices])
    feature_importance_dir = os.path.join(output_pth, "feature_importance.png")
    plt.xticks(range(test_features.shape[1]), names, rotation=90)
    plt.savefig(feature_importance_dir)
    plt.close()


def classification_report_plot(train_outputs,
                               pred_train,
                               test_outputs,
                               pred_test,
                               name_model,
                               output_pth="images/results/"):
    """
    Create classification report given and stores the figure.
    input:
        train_outputs: Train response values
        pred_train: Predictions based on train inputs
        test_outputs: Test response values
        pred_test: Predictions based on test inputs
        name_model: Name of the model included in the figure
        output_pth: output of model

    output: 
        None

    """
    # create folder for storing shap values figure
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)

    # Create classification report figure dir
    class_report_dir = os.path.join(output_pth,
                                    f"{name_model}_classification_report.png")
    # plot random forest report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str(f'{name_model} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(test_outputs, pred_test)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f'{name_model} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(train_outputs, pred_train)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    # Save figure
    plt.axis('off')
    plt.savefig(class_report_dir)
    plt.close()


if __name__ == "__main__":
    churn_df = import_data("data/BankChurners.csv")
    perform_eda(churn_df)
    trainX, testX, trainY, testY = perform_feature_engineering(churn_df)
    rf_model, lr_model, predictions = train_models(
        trainX, testX, trainY, testY)
    train_predict_rf, test_predict_rf, train_predict_lr, test_predict_lr = predictions
    classification_report_image(
        trainY,
        testY,
        testX,
        predictions,
        rf_model,
        lr_model)
    feature_importance_plot(rf_model, testX)
    classification_report_plot(
        trainY,
        train_predict_rf,
        testY,
        test_predict_rf,
        "Random Forest")
    classification_report_plot(
        trainY,
        train_predict_lr,
        testY,
        test_predict_lr,
        "Logistic Regression")
