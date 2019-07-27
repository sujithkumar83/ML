# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 17:05:11 2019
Decision Trees
@author: Sujith Kumar
"""

# Import librarires

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Data

dset= pd.read_csv('Position_Salaries.csv')
X=dset.iloc[:,1:2].values
Y=dset.iloc[:,2].values
#Y=Y.reshape(-1,1)

# Fitting Decision tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)


#predicting a new result
y_pred=regressor.predict(np.array([[6.5]]))


# Visualise the Result
xgrid=np.arange(min(X), max(X),0.01)
xgrid=xgrid.reshape(len(xgrid),1)
plt.scatter(X,Y, color="Red")
plt.plot(xgrid,regressor.predict(xgrid),color='Blue')
plt.title("Decision Tree Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()