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
X=dataset.iLoc[:,]

from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN', strategy='mean', axis=0)
imp=imputer.fit(X[])