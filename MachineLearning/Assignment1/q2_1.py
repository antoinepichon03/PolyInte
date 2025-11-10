import numpy as np
from q1_1 import rmse, ridge_regression_optimize, data_matrix_bias


# Part (a)
def cv_splitter(X, y, k):
    """
    Splits data into k folds for cross-validation.
    Returns a list of tuples: (X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    """

    n = X.shape[0]

    lignes_by_folds = n // k
    reste = n % k

    X_folds = []
    y_folds = []

    #Shuffle the data
    indices = np.arange(n)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]


    debut = 0 

    for i in range(k) :
        size = lignes_by_folds + (1 if i < reste else 0)
        fin = debut + size
        X_folds.append(X[debut:fin])
        y_folds.append(y[debut:fin])
        debut = fin
    
    X_train_folds = []
    y_train_folds = []
    X_val_folds = []
    y_val_folds = []

    for i in range(k):


        X_val_folds.append(X_folds[i])
        y_val_folds.append(y_folds[i])

        if len(X_folds) > 1:   
            X_train = np.vstack([X_folds[j] for j in range(len(X_folds)) if j != i])
            y_train = np.hstack([y_folds[j] for j in range(len(y_folds)) if j != i])
        else:
            X_train = X_folds[0]
            y_train = y_folds[0]

        X_train_folds.append(X_train)
        y_train_folds.append(y_train)
    
    fold = []
    for i in range(k):
        fold.append((X_train_folds[i], y_train_folds[i], X_val_folds[i], y_val_folds[i]))
    return fold



# Part (b)
def MAE(y, y_hat):
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    Y = np.abs(y - y_hat)
    mae = np.mean(Y)
    return mae



def MaxError(y, y_hat):
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    Y = np.abs(y - y_hat)
    maxError = np.max(Y)
    return maxError

#Importation du RMSE écrit en question 1
from q1_1 import (rmse,
                  ridge_regression_optimize,
                  data_matrix_bias)



# Part (c)


def cross_validate_ridge(X, y, lambda_list, k, metric):
    """
    Performs k-fold CV over lambda_list using the given metric.
    metric: one of "MAE", "MaxError", "RMSE"
    Returns the lambda with best average score and a dictionary of mean scores.
    """

    fold = cv_splitter(X,y,k)


    mean_scores = {}

    for labda in lambda_list:
        scores = []
        for i in range(k):
            X_train,  y_train, X_val,  y_val = fold[i]
            w = ridge_regression_optimize(data_matrix_bias(X_train), y_train, labda)
            y_hat = data_matrix_bias(X_val) @ w
            y_hat = y_hat.flatten()
            score = 0

            if metric == "MAE":
                score = MAE(y_val, y_hat)   
            
            elif metric == "MaxError" : 
                score = MaxError(y_val, y_hat)
            
            elif metric == "RMSE":
                score = rmse(y_val, y_hat)

            else :
                raise ValueError("Metric not recognized : not in the following list : MAE, MaxError, RMSE")

            scores.append(score)
        
        mean_scores[labda] = np.mean(scores)

    
    best_lambda = min(mean_scores, key=mean_scores.get)

    return best_lambda, mean_scores
            



        