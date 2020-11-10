# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Feature variables
x = dataset.iloc[:, 1:-1].values

# Dependency variable
y = dataset.iloc[:, -1].values

# Training the linear regression model on the whole dataset for comparison reasons
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Training the polynomial regression model on the whole dataset
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the linear regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the polynomial regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results(for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
y_pred1 = lin_reg.predict([[6.5]])
print(y_pred1)

# Predicting a new result with Polynomial Regression
y_pred2 = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(y_pred2)