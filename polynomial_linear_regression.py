# I. Data Preprocessing

# 1. Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Polynomial Linear Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

# Visualizing the testset results of Simple Linear Regression
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg.predict(X), color ='blue')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Salary')
plt.ylabel('Position')
plt.show()

# Visualizing the testset results of Polynomial Linear Regression
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color ='blue')
plt.title('Truth or bluff (Polynomial Linear Regression)')
plt.xlabel('Salary')
plt.ylabel('Position')
plt.show()

y_test = [[6.5]]
# Predicting a new result with Linear Regression
lin_reg.predict(y_test)
# Predicting a new result with Polynomial Linear Regression
lin_reg2.predict(poly_reg.fit_transform(y_test))