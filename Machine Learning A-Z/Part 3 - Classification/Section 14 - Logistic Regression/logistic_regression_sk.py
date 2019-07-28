# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:37:27 2019

Logisitic Regression Tutorial

@author: Sujith Kumar
"""
#Import Lbraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Data

dset=pd.read_csv('Social_Network_Ads.csv')
X=dset.iloc[:,2:4].values
Y=dset.iloc[:,4].values

#Split into Test and Train
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(X,Y,test_size=.25, random_state=0)
#ytrain=ytrain.reshape(-1,1)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
scalerx=StandardScaler()
xtrain=scalerx.fit_transform(xtrain)
xtest= scalerx.transform(xtest)
# =============================================================================
#not needed 
#scalery=StandardScaler()
# ytrain=scalery.fit_transform(ytrain)
# =============================================================================


#Fit Regressor
from sklearn.linear_model import LogisticRegression
regressor= LogisticRegression(random_state=0)
regressor.fit(xtrain,ytrain)

#Predicting 
y_pred=regressor.predict(xtest)

#Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true=ytest,y_pred=y_pred)

#Plotting graphs- Training Set
from matplotlib.colors import ListedColormap
xset, yset= xtrain, ytrain
x1, x2= np.meshgrid(np.arange(start=xset[:,0].min()-1,stop=xset[:,0].max()+1,step=.01),
                    np.arange(start=xset[:,1].min()-1,stop=xset[:,1].max()+1,step=.01))
plt.contourf(x1,x2,regressor.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape), alpha=.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset==j, 0], xset[yset==j, 1],
                c=ListedColormap(('red', 'green'))(i),label=j)
    plt.title('Logisitic Regression- Training Set')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend
    plt.show()
    
#Plotting graphs- Test Set
from matplotlib.colors import ListedColormap
xset, yset= xtest, ytest
x1, x2= np.meshgrid(np.arange(start=xset[:,0].min()-1,stop=xset[:,0].max()+1,step=.01),
                    np.arange(start=xset[:,1].min()-1,stop=xset[:,1].max()+1,step=.01))
plt.contourf(x1,x2,regressor.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape), alpha=.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset==j, 0], xset[yset==j, 1],
                c=ListedColormap(('red', 'green'))(i),label=j)
    plt.title('Logisitic Regression- Test Set')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend
    plt.show()
    