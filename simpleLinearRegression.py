#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv("headbrain.csv")
# print(data.shape)
# print(data.head())

#collecting X and Y
X = data["Head Size(cm^3)"].values
Y = data["Brain Weight(grams)"].values

#Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)
#print(mean_x, mean_y)

# or (without function mean() ):
# X_array = np.array(X)
# Y_array = np.array(Y)
# X_mean = sum(X_array) / len(X_array)
# Y_mean = sum(Y_array) / len(Y_array)
# print(X_mean, Y_mean)

d_x = np.array(X) - mean_x
d_y = np.array(Y) - mean_y

b1 = sum(d_x * d_y) / sum(np.power(d_x, 2))
b0 = mean_y - b1 * mean_x
BrainWeight = b0 + b1
#print(b1, b0)

#plotting values and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100

#Calulating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

#Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
#Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

# plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
#plt.show()

m = len(X)
rmse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse / m)
print(rmse)

ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r / ss_t)
print(r2)
