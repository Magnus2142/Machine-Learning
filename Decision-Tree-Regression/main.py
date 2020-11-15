# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Import dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Feature variables
x = dataset.iloc[:, 1:-1].values
# Dependency variable
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

# Predicting new results
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualizing the results

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
