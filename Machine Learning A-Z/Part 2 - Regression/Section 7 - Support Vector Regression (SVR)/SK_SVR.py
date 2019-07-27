# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:20:04 2019

@author: Sujith Kumar
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
dset= pd.read_csv('Position_Salaries.csv')
X=dset.iloc[:,1:2].values
Y=dset.iloc[:,2].values
Y=Y.reshape(-1,1)
# Apply Feature Scaling
from sklearn.preprocessing import StandardScaler
scalerx=StandardScaler()
X=scalerx.fit_transform(X)
scalery=StandardScaler()
Y=scalery.fit_transform(Y)

# Fit Regressor
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,Y)


#predicting a new result
y_pred=scalery.inverse_transform(regressor.predict(scalerx.transform(np.array([[6.5]]))))

# Visualise the Result
plt.scatter(X,Y, color="Red")
plt.plot(X,regressor.predict(X),color='Blue')
plt.title("Turth or Bluff")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()