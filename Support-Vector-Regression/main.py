# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Import dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Feature variables
x = dataset.iloc[:, 1:-1].values
# Dependency variable
y = dataset.iloc[:, -1].values

print(x)
print(y)

y = y.reshape(len(y), 1)

# Feature Scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print(x)
print(y)

# Training the SVR model on the whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(x, y.ravel())


# Predicting a new result
x_test_scaled = sc_x.transform([[6.5]])
y_pred = sc_y.inverse_transform(regressor.predict(x_test_scaled))
print(y_pred)


# Visualising the SVR results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color='blue')
plt.title('Truth or Bluff (Support Vector Regression))')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results smoothly
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color='blue')
plt.title('Truth or Bluff SMOOTH (Support Vector  Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



