from q2_1 import *
import pandas as pd
import numpy as np

# Write your code here ...
# Not autograded — function names and structure are flexible.

lambda_list = [0.01, 0.1, 1, 10, 100]

# Etape 0, chargement des données
X_test = np.loadtxt("X_test.csv", delimiter=",",  skiprows=1)
y_test = np.loadtxt("y_test.csv", delimiter=",",  skiprows=1)
X_train = np.loadtxt("X_train.csv", delimiter=",",  skiprows=1)
y_train = np.loadtxt("y_train.csv", delimiter=",",  skiprows=1)



liste_metric = ["RMSE", "MAE", "MaxError"]
# Etape 1  on cherche le meilleur lambda par validation croisée pour k = 5 et pour toutes les métric

result = []
for metric in liste_metric:
    best_lambda, mean_scores = cross_validate_ridge(X_train, y_train, lambda_list, 5, "RMSE")
    result.append((metric, best_lambda, mean_scores[best_lambda]))


#Etape 3 : on affiche le résultat
for metric, best_lambda, score in result:
    print(f"Metric : {metric}, Best lambda : {best_lambda}, Score : {score:.2f}")


