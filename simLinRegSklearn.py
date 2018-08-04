#!/usr/bin/env Python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("headbrain.csv")
# print(data.shape)
# print(data.head())

#collecting X and Y
X = data["Head Size(cm^3)"].values
Y = data["Brain Weight(grams)"].values

m = len(X)

#cannot use Rank 1 matrix in scikit learn
X = X.reshape((m, 1))
#creating Model
reg = LinearRegression()
#fitting training data
reg = reg.fit(X, Y)
#Y prediction
Y_pred = reg.predict(X)

#Calculating RMSE and R2 Score
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, Y)

print(np.sqrt(mse))
print(r2_score)