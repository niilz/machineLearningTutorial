#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("student.csv")
# print(data.shape)
# print(data.head())

math = data["Math"]
read = data["Reading"]
write = data["Writing"]

#X and Y values
X = np.array([math, read]).T
Y = np. array(write)

#Model Initialization
reg = LinearRegression()
#Data Fitting
reg = reg.fit(X, Y)
#Y prediction
Y_pred = reg.predict(X)
print(Y_pred)

#Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)

print(rmse)
print(r2)