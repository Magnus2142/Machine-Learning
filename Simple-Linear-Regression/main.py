# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import dataset
dataset = pd.read_csv("Salary_Data.csv")

# Feature variables
x = dataset.iloc[:, :-1].values

# Dependency variable
y = dataset.iloc[:, -1].values


# Splitting the dataset into the training the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the test set results
y_pred = regressor.predict(x_test)

# Visualize the results!

# Training set results
plt.scatter(x_train, git ho, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title('Salary vs Experience (Training set)')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

# Test set results
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title('Salary vs Experience (Test set)')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()