# train_model.py
import os
import warnings
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score, classification_report

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the preprocessing functions from build_features.py
from src.features.build_features import preprocess_numeric_data, preprocess_categorical_data, combine_processed_data

def split_train_test(X, y, size=0.1, seed=42):
    xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=size, random_state=seed, stratify=y)
    xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=size / (1 - size), random_state=seed, stratify=ytrain)
    return xtrain, xtest, xval, ytrain, ytest, yval

def eval_metrics(yval, ypred):
    cm = confusion_matrix(yval, ypred)
    preci = precision_score(yval, ypred)
    acc = accuracy_score(yval, ypred)
    f1_sc = f1_score(yval, ypred)
    cl_report = classification_report(y_true=yval, y_pred=ypred)
    return cm, preci, acc, f1_sc, cl_report

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    filepath = r"C:\Users\yazid\Desktop\Bank churn\Data\BCC.csv"
    try:
        df = pd.read_csv(filepath, sep=',')

        # Drop customer_id and churn columns
        X = df.iloc[:, 1:-1]
        Y = df.iloc[:, -1]

        xtrain, xtest, xval, ytrain, ytest, yval = split_train_test(X, Y, 0.15)

        numerics = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
        categorical_cols = ['country', 'gender', 'credit_card', 'active_member']

        xtrain_num, xtest_num, xval_num = preprocess_numeric_data(numerics, xtrain, xtest, xval)
        xtrain_cat, xtest_cat, xval_cat = preprocess_categorical_data(categorical_cols, xtrain, xtest, xval)

        xtrain_scl, xtest_scl, xval_scl = combine_processed_data((xtrain_num, xtest_num, xval_num), (xtrain_cat, xtest_cat, xval_cat))

        reg = 0.001

        # Create and save polynomial features
        poly = PolynomialFeatures(degree=3, include_bias=True)
        x_poly = poly.fit_transform(np.asarray(xtrain_scl))
        joblib.dump(poly, "src/models/polynomial_features.joblib")

        # Train and save logistic regression model
        poly_reg_model = LogisticRegression(class_weight={0: 1, 1: 1}, C=reg)
        poly_reg_model.fit(x_poly, ytrain)
        joblib.dump(poly_reg_model, "src/models/model_logistic.joblib")

        ypred_poly = poly_reg_model.predict(poly.transform(np.asarray(xval_scl)))

        cm, preci, acc, f1_sc, cl_report = eval_metrics(yval, ypred_poly)

        print("Polynomial Features + Logistic regression model (C={:f}):".format(reg))
        print("  Accuracy: %s" % acc)
        print("  Precision: %s" % preci)
        print("  f1-score: %s" % f1_sc)
        print("  Confusion Matrix:\n %s" % cm)

    except Exception as e:
        print("Exception occurred: ", e)
