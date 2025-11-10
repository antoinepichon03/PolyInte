import numpy as np
from scipy.stats import multivariate_normal
import typing


# -----------------------
# Gaussian Naive Bayes
# -----------------------
def gnb_fit_classifier(X: np.ndarray, Y: np.ndarray, smoothing: float = 1e-3) -> typing.Tuple:
    """
    Fits the GNB classifier on the training data
    """
    prior_probs = []
    means = []
    vars_ = []

    classes = np.unique(Y)
    # Your implementation here
    for k in classes : 
        X_k = X[Y == k]
        N_k = X_k.shape[0]
        N_total = X.shape[0]
            
        prior_k = N_k / N_total
        prior_probs.append(prior_k)
            
        mean_k = np.mean(X_k, axis=0)
        means.append(mean_k)
        
        var_k = np.var(X_k, axis=0) + smoothing 

        vars_.append(var_k)
    return prior_probs, means, vars_


def gnb_predict(
    X: np.ndarray,
    prior_probs: typing.List[float],
    means: typing.List[np.ndarray],
    vars_: typing.List[np.ndarray],
    num_classes: int,
) -> np.ndarray:
    """
    Computes predictions from the GNB classifier
    """
    N_test = X.shape[0]
    log_probs = np.zeros((N_test, num_classes))
    preds = None
    # Your implementation here

    
    
    for k in range(num_classes):
        # Récupérer éléments de la classe k
        prior_k = prior_probs[k]
        mean_k = means[k]
        var_k = vars_[k]
        #Dans notre cas, on a 
        # log[N(x|mu,sigma^2)] =-1/2 *[log(2 pi sigma^2)+(x-mu)^2/sigma^2]
        diff = X - mean_k 
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var_k) + (diff**2) / var_k, axis=1)
        log_probs[:, k] = log_likelihood +  np.log(prior_k)
    
    # Prédictions est la classe avec la plus grande probabilité en log, fonction croissante
    preds = np.argmax(log_probs, axis=1)
    return preds


def gnb_classifier(train_set, train_labels, test_set, test_labels, smoothing=1e-3):
    """
    Runs GNB classifier and computes accuracy
    """
    num_classes = len(np.unique(train_labels))
    priors, means, vars_ = gnb_fit_classifier(train_set, train_labels, smoothing)
    y_pred = gnb_predict(test_set, priors, means, vars_, num_classes)
    accuracy = np.mean(y_pred == test_labels) * 100.0
    return accuracy


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    from data_process import preprocess_mnist_data
    import numpy as np

    # MNIST dataset (from CSVs prepared by data_download.py)
    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist, _, _ = preprocess_mnist_data(
        "data/MNIST/mnist_train.csv", "data/MNIST/mnist_test.csv"
    )

    print("Evaluating on MNIST...")
    gnb_acc_mnist = gnb_classifier(X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist)
    print(f"MNIST - GNB accuracy: {gnb_acc_mnist:.2f} %")

    # IRIS dataset (CSV created by data_download.py): last column is label
    train_iris = np.loadtxt("data/iris/iris_train.csv", delimiter=",")
    test_iris = np.loadtxt("data/iris/iris_test.csv", delimiter=",")
    X_train_iris, y_train_iris = train_iris[:, :-1], train_iris[:, -1].astype(int)
    X_test_iris, y_test_iris = test_iris[:, :-1], test_iris[:, -1].astype(int)

    print("\nEvaluating on IRIS...")
    gnb_acc_iris = gnb_classifier(X_train_iris, y_train_iris, X_test_iris, y_test_iris)
    print(f"IRIS - GNB accuracy: {gnb_acc_iris:.2f} %")

    # Taux d'erreur pour chacune des classes 

    nombre_classe = len(np.unique(y_train_mnist))
    priors, means, vars_ = gnb_fit_classifier(X_train_mnist, y_train_mnist)
    y_pred_mnist = gnb_predict(X_test_mnist, priors, means, vars_, nombre_classe)
    for i in range(10):
        mask = y_test_mnist == i

        if np.sum(mask) > 0:
            class_accuracy = np.mean(y_pred_mnist[mask] == y_test_mnist[mask])*100
            print(f"Accuracy de la classe {i}: {class_accuracy:.2f} %")


# # -----------------------
# # Quadratic Discriminant Analysis # You are lucky you don't have to do anything about this!
# # -----------------------
# def qda_fit_model(X: np.ndarray, Y: np.ndarray, reg: float = 1e-3) -> typing.Tuple:
#     """
#     Fit QDA model: compute mu_k and full covariance Sigma_k per class
#     """
#     priors, means, covariances = [], [], []
#     return priors, means, covariances


# def qda_predict(
#     X: np.ndarray, priors: typing.List[float], means: typing.List[np.ndarray], covariances: typing.List[np.ndarray]
# ) -> np.ndarray:
#     """
#     Computes predictions from a QDA classifier
#     """
#     log_probs = None
#     preds = None
#     return preds


# def qda_classifier(train_set, train_labels, test_set, test_labels, reg=1e-3):
#     """
#     Run QDA classifier and return accuracy
#     """
#     priors, means, covariances = qda_fit_model(train_set, train_labels, reg)
#     y_pred = qda_predict(test_set, priors, means, covariances)
#     accuracy = np.mean(y_pred == test_labels) * 100.0
#     return accuracy
