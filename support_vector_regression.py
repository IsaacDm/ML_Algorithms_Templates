# I. Data Preprocessing

# 1. Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.array(y).reshape(-1,1)
y = sc_y.fit_transform(y)
y = y.flatten()
# Support Vector Regression

# 1. Fitting the Support Vector Regression to the Dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

# 2. Predicting the test results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# 3. Visualizing the training set results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('SVR')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()