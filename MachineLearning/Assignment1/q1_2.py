import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q1_1 import (
    data_matrix_bias,
    linear_regression_optimize,
    ridge_regression_optimize,
    weighted_ridge_regression_optimize,
    predict,
    rmse
)

# Write your code here ...
# Not autograded — function names and structure are flexible.

lambda_vec = [ 0.01, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3]
# Etape 0, chargement des données
X_test = np.loadtxt("X_test.csv", delimiter=",",  skiprows=1)
y_test = np.loadtxt("y_test.csv", delimiter=",",  skiprows=1)
X_train = np.loadtxt("X_train.csv", delimiter=",",  skiprows=1)
y_train = np.loadtxt("y_train.csv", delimiter=",",  skiprows=1)

# Etape 1, ajouter la colone de biais
X_test_bias = data_matrix_bias(X_test)
X_train_bias = data_matrix_bias(X_train)

#Etape 2, entrainement du modèle
w_ordinary = linear_regression_optimize(X_train_bias, y_train)
w_ridge = ridge_regression_optimize(X_train_bias, y_train, 0.1)
w_ridge_weighted = weighted_ridge_regression_optimize(X_train_bias, y_train, lambda_vec)

#Etape 3, RMSE sur les test
y_hat_ordinary = predict(X_test_bias, w_ordinary)
y_hat_ridge = predict(X_test_bias, w_ridge)
y_hat_weight_ridge = predict(X_test_bias, w_ridge_weighted)

rmse_ordinary = rmse(y_test, y_hat_ordinary)
rmse_ridge = rmse(y_test, y_hat_ridge)
rmse_weight_ridge = rmse(y_test, y_hat_weight_ridge)

#plot
plt.figure()
plt.scatter(y_test, y_hat_ordinary, label = "Ordinary")
plt.scatter(y_test, y_hat_ridge, label = "ridge")
plt.scatter(y_test, y_hat_weight_ridge, label = "weight_ridge")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal")
plt.title(f"RMSE OLS: {rmse_ordinary:.2f}, Ridge: {rmse_ridge:.2f}, Weighted: {rmse_weight_ridge:.2f}")
plt.legend()
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()