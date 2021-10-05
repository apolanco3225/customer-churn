# Predict Customer Churn
### First Projec Machine Learning Engineer DevOps Nanodegree

In this project, the job is to identify credit card customers that are most likely to churn. This repository includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package has the flexibility of being run interactively or from the command-line interface (CLI).

The data used is called Credit Card Customers, you can download it from [this link](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code)

The skeleton project included a [Jupyter Notebook file with the data loading, data exploration, data segregation, model training and testing](https://github.com/apolanco3225/customer-churn/blob/main/churn_notebook.ipynb). In the following files I tried to improve this process, making it more adequate for a production environment.

This project contains:
* churn_library.py: Modular and reproducible.
* churn_script_logging_and_tests.py: Making sure tests are added in the model pipeline.

## Running Files
Please create an environment with ***Python 3.8***, then proceed to install all the requirements using 
```
pip install -r requirements.txt
```
Once all the depndencies are installed you need to exectute in your terminal:
```
python churn_library.py
```
This command will exectute the whole machine learning pipeline, it will generate artifacts for:
- The EDA in the image/eda folder with information of the Churn dataframe.
- Trained Models of Logistic Regression and Random Fores in the models folder.
- The Testing in the image/results folder with information of classification report, AUC for Random Forest and Logistic Regression classifiers, shap values for Random Forest and Feature Importance. 

Test that the pipeline is working correctly by executing:
```
python churn_script_logging_and_tests.py 
```
This command will generate the following artifact:
- Log file with the results of all the tests applied in the log folder.
```
```
