#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Apr 11 16:04:55 2018

@author: ChenFangjie

Multiple Linear Regression
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Taking care of missing data


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling


# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# 在 X_train 中加入常数列，因为 下面用到的sm.OLS函数不包含常数
X_train_row_size = np.shape(X_train)[0];
X_train = np.append(arr = np.ones((X_train_row_size, 1)), values = X_train, axis = 1)

# =============================================================================
# X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# regressor_OLS.summary()
# =============================================================================

# =============================================================================
# X_opt = X_train[:, [0, 1, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# regressor_OLS.summary()
# =============================================================================

# =============================================================================
# X_opt = X_train[:, [0, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# regressor_OLS.summary()
# =============================================================================

X_opt = X_train[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

# =============================================================================
# X_opt = X_train[:, [0, 3]]
# regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# regressor_OLS.summary()
# =============================================================================


X_train_2 = X_train[:, [3, 5]]
X_test_2 = X_test[:, [2, 4]]

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_2, y_train)

# Predicting the Test set results
y_pred_2 = regressor.predict(X_test_2)
