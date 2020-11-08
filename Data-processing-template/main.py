# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import dataset
dataset = pd.read_csv("Data.csv")

# Feature variables
x = dataset.iloc[:, :-1].values

# Dependency variable
y = dataset.iloc[:, -1].values

# Taking care of rows with missing data

# deciding how to identify missing values (nan) and how to replace them (mean)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# give the imputer the columns where the missing values are
imputer.fit(x[:, 1:3])
# imputer replaces the missing values and returns the new columns with all values + the replaced ones.
x[:, 1:3] = imputer.transform(x[:, 1:3])


# Encoding categorical data

# encode the independent data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# encode the dependant data
le = LabelEncoder()
y = le.fit_transform(y)


# Splitting the dataset into the training the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# Feature scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)
