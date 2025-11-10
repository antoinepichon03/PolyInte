import numpy as np
from typing import Tuple

np.random.seed(42)


# -----------------------
# Forward probabilities
# -----------------------
def softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities for each row of z
    """

    # Implement here
    z -= np.max(z, axis=1, keepdims=True) 
    result = np.exp(z)/(np.sum(np.exp(z), axis=1, keepdims=True))
    return result


def forward_probabilities(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes class probabilities P(Y|X) given parameters W, b
    """
    # Implement here
    logits = X@W+b
    results = softmax(logits)
    return results  # shape: [num_samples, num_classes]


# -----------------------
# Logistic regression loss
# -----------------------
def logistic_regression_loss(
    X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, reg_lambda: float = 0.0
) -> float:
    """
    Computes cross-entropy loss with optional L2 regularization
    """
    N = X.shape[0]
    prob = forward_probabilities(X,W,b)
    log_prob_correct = np.log(prob[np.arange(N), y]+ 1e-12)
    loss = -(1/N) * np.sum(log_prob_correct)
    loss += 0.5 * reg_lambda * np.sum(W * W)  # We add a L2 regularization term here, do not remove it
    return loss


# -----------------------
# Logistic regression gradients
# -----------------------
def logistic_regression_grad(
    X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, reg_lambda: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes gradient of cross-entropy loss w.r.t W and b
    """
    N, K = X.shape[0], W.shape[1]
    probs = forward_probabilities(X, W, b)
    yij = np.zeros((N, K))
    yij[np.arange(N), y] = 1

    error = probs - yij
    
    #Calcul des gradients
    grad_W = (1/N) * (X.T @ error) 
    grad_b = (1/N) * np.sum(error, axis=0)  
    grad_W += reg_lambda * W
    grad_b += 0
    return grad_W, grad_b


# -----------------------
# Logistic regression fit
# -----------------------
def logistic_regression_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    lr: float = 0.1,
    num_iters: int = 100,
    reg_lambda: float = 0.0,
    val_every: int = 10,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Train logistic regression using gradient descent
    """
    num_features = X_train.shape[1]
    W = np.random.randn(num_features, num_classes) * 0.01
    b = np.zeros(num_classes)

    val_acc_list = []
    val_iters = []

    for it in range(1, num_iters + 1):
        grad_W, grad_b = logistic_regression_grad(X_train, y_train, W, b, reg_lambda)
        W -= lr * grad_W
        b -= lr * grad_b

        if it % val_every == 0:
            probs_val = forward_probabilities(X_val, W, b)
            y_pred_val = np.argmax(probs_val, axis=1)
            acc_val = np.mean(y_pred_val == y_val)
            val_acc_list.append(acc_val)
            val_iters.append(it)
            print(f"Iteration {it}: Validation accuracy = {acc_val:.4f}")

    return W, b, val_iters, val_acc_list


# -----------------------
# Prediction
# -----------------------
def predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    probs = forward_probabilities(X, W, b)
    return np.argmax(probs, axis=1)


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    from data_process import preprocess_credit_card

    # Load credit card fraud dataset (same paths as KD-tree KNN example)
    X_train, y_train, X_test, y_test, mean, std = preprocess_credit_card(
        "data/credit_card_fraud/credit_card_fraud_train.csv",
        "data/credit_card_fraud/credit_card_fraud_test.csv",
    )

    # Create a validation split from the training set
    rng = np.random.RandomState(42)
    n_train = X_train.shape[0]
    val_size = max(1, int(0.2 * n_train))
    perm = rng.permutation(n_train)
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train_split, y_train_split = X_train[train_idx], y_train[train_idx]

    num_classes = len(np.unique(y_train_split))

    # Train logistic regression with different regularization parameters
    reg_values = [0, 0.01, 0.1, 1, 10]
    results = {}
    for reg in reg_values:
        print(f"\nTraining with lambda = {reg}")
        W_trained, b_trained, val_iters, val_acc_list = logistic_regression_fit(
            X_train_split,
            y_train_split,
            X_val,
            y_val,
            num_classes,
            lr=0.1,
            num_iters=100,
            reg_lambda=reg,
            val_every=10,
        )
        results[reg] = (W_trained, b_trained, val_iters, val_acc_list)

    # Pick best lambda (highest final validation accuracy)
    best_lambda = max(results, key=lambda k: results[k][3][-1])
    W_best, b_best = results[best_lambda][:2]

    # Evaluate on test set
    y_test_pred = predict(X_test, W_best, b_best)
    test_acc = np.mean(y_test_pred == y_test)
    print(f"\nBest lambda: {best_lambda}, Test accuracy: {test_acc:.4f}")


    #CODE AJOUTE
    #On ajoute le plot demandé dans l'éononcé
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    for reg in reg_values:
        W_trained, b_trained, val_iters, val_acc_list = results[reg]
        plt.plot(val_iters, val_acc_list, marker='o', label=f'lambda = {reg}')
    plt.title("Accuracy en fonction des itérations pour différentes valeurs de lambda")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
