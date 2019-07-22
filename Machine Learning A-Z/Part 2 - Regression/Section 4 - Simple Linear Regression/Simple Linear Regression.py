# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:47:44 2019
@author: Sujith Kumar
"""
#Definign Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Data

dset= pd.read_csv('Salary_Data.csv')

#Define X and Y
X= dset.iloc[:,:-1].values
Y=dset.iloc[:,1].values

#Missing Values Imputations- not working for some reason
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='mean', axis=)
imp=imp.fit(X[:,0])
X[0]=imp.transform(X[0])
Y[:,0]=imp.fit_transform(Y[:,0])


#Creating Train and Test datasets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)


#Fitting Simple Linear Regression

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(X_train, Y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)


#Visualising the Training Set Results

plt.scatter(X_train,Y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test Set Results

plt.scatter(X_test,Y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


