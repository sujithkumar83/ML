#Polynomial Regression Example

#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
dset=pd.read_csv('Position_Salaries.csv')

print(type(dset))
#Divide the dataset into Dependent and Indepenedent variables
X=dset.iloc[:,1:2].values
Y=dset.iloc[:,-1].values

#Fitting Linear Remression model
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,Y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg=PolynomialFeatures(degree=4)
xpoly=polyreg.fit_transform(X)
linreg2=LinearRegression()
linreg2.fit(xpoly,Y)

#Visualising Linear Regression Results
plt.scatter(X,Y,color='Red')
plt.plot(X, linreg.predict(X), color='Blue')
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")

#Visualising the polynomial Regression Results
xgrid=np.arange(min(X),max(X),0.1)
xgrid=xgrid.reshape(len(xgrid),1)
plt.scatter(X,Y,color='Red')
plt.plot(xgrid, linreg2.predict(polyreg.fit_transform(xgrid)), color='Blue')
plt.title("Linear Regression")
plt.xlabel("Position Level Squared")
plt.ylabel("Salary")

#Visualising the polynomial Regression Results


