import numpy as np


# Part (a)
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias column of ones as the first column of X."""
    X = np.asarray(X)
    n = X.shape[0]
    bias_colonne = np.ones((n,1))
    new_x = np.hstack((bias_colonne, X))
    return new_x


# Part (b)
def linear_regression_optimize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS solution"""
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.linalg.inv(X.T @ X) @ X.T @ y 
    return w.flatten()

# Part (c)
def ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lamb: float) -> np.ndarray:
    """Closed-form Ridge regression."""
    X = np.asarray(X)
    y = np.asarray(y)
    m=X.shape[1]
    Id = np.eye(m)
    w = np.linalg.inv(X.T @ X + lamb*Id) @ X.T @ y
    return w.flatten()

# Part (e)
def weighted_ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lambda_vec: np.ndarray) -> np.ndarray:
    """Weighted Ridge regression solution."""
    X = np.asarray(X) 
    y = np.asarray(y)
    lambda_vec = np.asarray(lambda_vec)
    A = np.diag(lambda_vec)
    w = np.linalg.inv(X.T @ X + A) @ X.T @ y
    return w.flatten()

# Part (f)
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute predictions: y_hat = X w"""
    X = np.asarray(X) 
    w = np.asarray(w)
    y_hat = X@w
    return y_hat.flatten()

# Part (f)
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error"""
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    Y_centre = y - y_hat
    rmse = np.sqrt(np.mean(Y_centre**2))
    return rmse

