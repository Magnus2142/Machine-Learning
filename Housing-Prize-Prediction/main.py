# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Import dataset
dataset = pd.read_csv("kc_house_data.csv")

# Feature variables
x = dataset.iloc[:, dataset.columns != 'price'].values


# Turn date into number
for date in range(0, len(x[:, 1])):
    x[date, 1] = x[date, 1].replace('T000000', '')

# Dependency variable
y = dataset.iloc[:, 2].values


# Splitting the dataset into the training the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# Set max decimal output to two digits
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))



