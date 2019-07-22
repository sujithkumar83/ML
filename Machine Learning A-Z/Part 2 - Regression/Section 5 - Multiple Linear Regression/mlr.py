# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:41:29 2019

Multiple Linear Regression Example

@author: Sujith Kumar
"""

#Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Import Data

dset= pd.read_csv('50_Startups.csv')
X=dset.iloc[:,:-1].values
Y=dset.iloc[:,-1].values

"""
#Approaches for MLR

**All (var) in
**Forward Selection
**Backward Selection
**Bidirectional Selection
**All possible models

"""
#encoding Categorical variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
lblencdr=LabelEncoder()
X[:,3]=lblencdr.fit_transform(X[:,3])
onhtenc=OneHotEncoder(categorical_features=[3])
X=onhtenc.fit_transform(X).toarray()

#Avoiding the dummy variable trap. You only need n-1 columns for specify n cat variations 



#Split the dataset into Training and Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Fit Regressor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test Set Results
y_pred=regressor.predict(X_test)


#Build Up a model using Backward Elimination
import statsmodels.formula.api as sm

X= np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,1,2,3,4]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()



