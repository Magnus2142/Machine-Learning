# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

# Import dataset
dataset = pd.read_csv("50_Startups.csv")

# Feature variables
x = dataset.iloc[:, :-1].values
# Dependency variable
y = dataset.iloc[:, -1].values

# Encoding categorical data

# encode the independent data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Avoiding the dummy variable trap
# NB! this is only necessary if you are not using the sklearn library for making
# the multiple linear regression line, because the sklearn avoids the dummy variable
# trap for you!
x = x[:, 1:]

# Splitting the dataset into the training the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ------------------------ Backward elimination with sklearn ------------------------ #

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predict the test set results
y_pred = regressor.predict(x_test)

# Set max decimal output to two digits
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# ------------------------ Backward elimination from scratch ------------------------ #
# Adds a column of 1's in the beginning which acts as the b0 constant.
x_train = np.append(arr=np.ones((len(x), 1)).astype(int), values=x, axis=1)

# Starts with all independent variables, finds the predictor with the
# highest p-value and checks if it is larger than our significance level (0.05)
x_opt = x_train.astype(int)[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

# column 2 had the highest and had a p-value larger than 0.05
x_opt = x_train.astype(int)[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

# column 1 had the highest and had a p-value larger than 0.05
x_opt = x_train.astype(int)[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

# column 4 had the highest and had a p-value larger than 0.05
x_opt = x_train.astype(int)[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

# column 5 had the highest and had a p-value larger than 0.05
x_opt = x_train.astype(int)[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

y_pred2 = regressor_OLS.predict(x_test[:, [0, 3]])

print(np.concatenate((y_pred2.reshape(len(y_pred2), 1), y_test.reshape(len(y_test), 1)), 1))



