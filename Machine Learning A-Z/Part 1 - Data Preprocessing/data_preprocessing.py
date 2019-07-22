# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Data Pre processing templates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset

dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#imputing missing values

from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN', strategy='mean', axis=0)
imp=imp.fit(X[:,1:3])
X[:,1:3]=imp.transform(X[:,1:3])
"""
Note:
imp=imp.fit(X[:,1:3])
X[:,1:3]=imp.transform(X[:,1:3])

is same as


X[:,1:3]=imp.fit_transform(X[:,1:3])
"""


#encoding Categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lblenc_X=LabelEncoder()
X[:,0]=lblenc_X.fit_transform(X[:,0])
onhtenc=OneHotEncoder(categorical_features=[0])
X=onhtenc.fit_transform(X).toarray()
lblenc_Y=LabelEncoder()
Y=lblenc_Y.fit_transform(Y)

#Splitting for training and out of test validation


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2, random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
X_train =sc_x.fit_transform(X_train)
X_test =sc_x.transform(X_test)

