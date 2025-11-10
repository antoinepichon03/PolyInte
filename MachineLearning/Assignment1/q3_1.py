import numpy as np


# Part (a)
def ridge_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb: float) -> np.ndarray:
    """
    Computes the gradient of Ridge regression loss.
    ∇L(w) = -2/n X^T (y - X w) + 2 λ w
    """
    n = X.shape[0]
    gradL = (-2/n) * X.T @ (y - X @ w) + 2 * lamb * w
    return gradL 


# Part (b)
def learning_rate_exp_decay(eta0: float, t: int, k_decay: float) -> float:
    eta = eta0 * np.exp(-k_decay * t)
    return eta


# Part (c)
def learning_rate_cosine_annealing(eta0: float, t: int, T: int) -> float:
    eta = eta0 * 0.5 * (1+ np.cos(np.pi * t / T))
    return eta


# Part (d)
def gradient_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb:float, eta: float) -> np.ndarray:
    grad = ridge_gradient(X, y, w, lamb)
    w_new = w - eta * grad
    return w_new


# Part (e)
def gradient_descent_ridge(X, y, lamb=1.0, eta0=0.1, T=500, schedule="constant", k_decay=0.01):
    n , m = X.shape

    #initialisation
    w= np.zeros(m)
    losses = []
    
    for t in range(T):
        if schedule == "constant":
            eta = eta0
        elif schedule == "exp_decay":
            eta = learning_rate_exp_decay(eta0, t, k_decay)
        elif schedule == "cosine":
            eta = learning_rate_cosine_annealing(eta0, t, T)
        else:
            raise ValueError("Unknown schedule type")
        w = gradient_step(X, y, w, lamb, eta)
        loss = (1/n) * np.sum((y - X @ w)**2) + lamb * np.sum(w**2)
        losses.append(loss)
    
    return w, losses



    

