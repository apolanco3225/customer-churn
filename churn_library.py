"""
This module predicts churn of credit card customers
Author: Arturo Polanco
Date: September 2021
"""

# import necessary libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth_path):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    dataframe = pd.read_csv(pth_path)
    return dataframe


def perform_eda(input_df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # exploratory data analysis
    print(input_df.head())
    print(input_df.shape)
    print(input_df.isnull().sum())
    print(input_df.describe())
    
    # create output Churn variable
    input_df['Churn'] = input_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # saving EDA figures
    input_df["Churn"].hist().figure.savefig("images/eda/churn_hist.png")
    input_df['Customer_Age'].hist().figure.savefig("images/eda/customer_age.png")
    input_df.Marital_Status.value_counts('normalize').plot(
        kind='bar').figure.savefig("images/eda/maritalStatus_count.png")
    sns.distplot(input_df['Total_Trans_Ct']).figure.savefig(
        "images/eda/distplot_total_trans_ct.png")
    sns.heatmap(input_df.corr(), annot=False, cmap='Dark2_r',
                linewidths=2).figure.savefig("images/eda/heatmap.png")


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
        for value in df[category]:
            column_list.append(grouped_values.loc[value])
        df[new_column_name] = column_list

    return df


def perform_feature_engineering(input_df, response=None):
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
    # set response value
    if response is None:
        response = "train"

    # create extra features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    input_df = encoder_helper(input_df, cat_columns, response=None)

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
    input_features[keep_cols] = input_df[keep_cols]
    # select churn column
    output_variable = input_df['Churn']

    # split the dataset for training and testing
    train_features, test_features, train_output, test_output = train_test_split(
        input_features, output_variable, test_size=0.3, random_state=42)

    return train_features, test_features, train_output, test_output


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Print classification report random forest
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    # print classification report logistic regression
    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    # store figure images ROC curve
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8).savefig("images/results/roc_curve.png")

    # store figure images AUC curve
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8).savefig("images/results/auc_iamge.png")
    

def feature_importance_plot(model, X_data, output_pth=None):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # calculate the shapley values
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # Calculate feature importances
    importances = cv_rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=90).savefig(
        "images/results/shap_values.png")

    # plot random forest report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off').savefig("images/results/shap_values.png")

    # plot logistic regression report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off').savefig("images/results/shap_values.png")
    

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
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

    # find best parameters for random forest and fit
    cv_rfc = GridSearchCV(
        estimator=random_forest_model,
        param_grid=param_grid,
        cv=5)
    cv_rfc.fit(X_train, y_train)

    # train logistic regression model
    logistic_refression_model.fit(X_train, y_train)

    # predict values of training and testing sets using best parameters for
    # random forest
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # predict values of training and testing sets using logistic regression
    y_train_preds_lr = logistic_refression_model.predict(X_train)
    y_test_preds_lr = logistic_refression_model.predict(X_test)

    # save best model random forest and logistic regression
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # load models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    

if __name__ == "__main__":
    df = import_data(r"./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, response=None)
    train_models(X_train, X_test, y_train, y_test)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    feature_importance_plot(cv_rfc, X_data, output_pth=None)
