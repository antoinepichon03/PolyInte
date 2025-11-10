import numpy as np
import scipy
import sklearn.metrics.pairwise as smp

from data_process import preprocess_mnist_data
from utils import visualize_image
import pandas as pd
import os


def euclidean_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the Euclidean distance between two arrays

    Args:
        A (np.ndarray): Numpy array of shape [num_samples_a x num_features]
        B (np.ndarray): Numpy array of shape [num_samples_b x num_features]

    Returns:
        np.ndarray: Numpy array of shape [num_samples_a x num_samples_b] where
                    each column contains the distance between one element in
                    matrix_b and all elements in matrix_a
    """
    distances = smp.euclidean_distances(A, B)

    ### Implement here
    # You might want to use a metric in sklearn.metrics.pairwise to avoid potential out-of-memory errors.
    # And to speed up the computation.

    return distances


def cosine_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the cosine distance between two arrays

    Args:
        A (np.ndarray): Numpy array of shape [num_samples_a x num_features]
        B (np.ndarray): Numpy array of shape [num_samples_b x num_features]

    Returns:
        np.ndarray: Numpy array of shape [num_samples_a x num_samples_b] where
                    each column contains the cosine distance between one element in
                    matrix_b and all elements in matrix_a
    """
    # NOTE: Similar to the euclidean_distance function, you might want to use
    # scikit-learn function to avoid potential out-of-memory errors.
    distances = smp.cosine_distances(A, B)
    ### Implement here
    return distances


def get_k_nearest_neighbors(distances: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Gets the k nearest labels based on the distances

    Args:
        distances (np.ndarray): Numpy array of shape num_train_samples x num_test_samples
                                containing the Euclidean distances
        labels (np.ndarray): Numpy array of shape [num_train_samples, ] containing
                                the training labels
        k (int): Number of nearest neighbours

    Returns:
        np.ndarray: Numpy array of shape [k x num_test_samples] containing the
                    training labels of the k nearest neighbours for each test sample
    """

    # Sort the distances in ascending and get the indices of the first "k" elements
    # HINT: You need to sort the distances in ascending order to get the indices
    # of the first "k" elements. BUT, you would not need to sort the entire array,
    # it would be enough to make sure that the "k"-th element is in the correct position!

    # NOTE: Since the matrix sizes are huge, it would be impractical to run any sort of a
    # loop to get the nearest labels. Think about how you can do it without using loops.

    
    ### Implement here, explain also briefly in a comment how you are doing it.
    ### If you use any external function, such as numpy's functions, please explain what the function does.



    # np.argpartition trouve les indices qui partitionnent l'array de sorte que 
    # les k premiers éléments soient les plus petits (sans tri complet)
    # axis=0 : on partition le long des lignes (échantillons d'entraînement)
    # kth=k-1 : le k-ème plus petit élément sera à la position k-1
    index = np.argpartition(distances, kth = k-1, axis=0)[:k, :]
    neighbors = labels[index]

    return neighbors


def majority_voting(nearest_labels: np.ndarray) -> np.ndarray:
    """Gets the best prediction, i.e. the label class that occurs most frequently

    Args:
        nearest_labels (np.ndarray): Numpy array of shape [k x num_test_samples] obtained from the output of the get_k_neighbors function

    Returns:
        np.array: Numpy array of shape [num_test_samples] containing the best prediction for each test sample. If there are more than one most frequent labels, return the smallest one.
    """



    ### Implement here

    # Utilisation de scipy.stats.mode pour trouver la valeur la plus fréquente
    # axis=0 signifie qu'on calcule le mode le long des colonnes (pour chaque échantillon test)
    # keepdims=False pour obtenir un array 1D en sortie
    try:
        # Version récente de scipy
        predicted = scipy.stats.mode(nearest_labels, axis=0, keepdims=False).mode
    except TypeError:
        # Version plus ancienne de scipy
        predicted = scipy.stats.mode(nearest_labels, axis=0).mode[0]
    
    # S'assurer que le résultat est un array 1D
    predicted = predicted.flatten()

    return predicted


def knn_classifier(
    training_set: np.ndarray,
    training_labels: np.ndarray,
    test_set: np.ndarray,
    test_labels: np.ndarray,
    k: int,
    dist_func: callable,
) -> float:
    """
    Performs k-nearest neighbour classification

    Args:
    training_set (np.ndarray): Vectorized training images (shape: [num_train_samples x num_features])
    training_labels (np.ndarray): Training labels (shape: [num_train_samples, 1])
    test_set (np.ndarray): Vectorized test images (shape: [num_test_samples x num_features])
    test_labels (np.ndarray): Test labels (shape: [num_test_samples, 1])
    k (int): number of nearest neighbours

    Returns:
    accuracy (float): the accuracy in %
    """
    # compute the distance between each test sample and all training samples
    # Cache distances per distance function and dataset pair to reuse across different k calls
    dists = dist_func(training_set, test_set)

    nearest_labels = get_k_nearest_neighbors(distances=dists, labels=training_labels, k=k)

    # from the nearest labels above choose the label classes that occurs most frequently
    predictions = majority_voting(nearest_labels)

    # calculate and return accuracy of the predicitions
    accuracy = (np.equal(predictions, test_labels).sum()) / len(test_set) * 100.0

    return accuracy


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, mean, std = preprocess_mnist_data(
        os.path.join(os.path.dirname(__file__), "data", "MNIST", "mnist_train.csv"),
        os.path.join(os.path.dirname(__file__), "data", "MNIST", "mnist_test.csv"),
    )

    
    #teste du code avec des données réduites : 
    #X_train = X_train[:3000]  
    #y_train = y_train[:3000]
    #X_val, y_val = X_train[-500:], y_train[-500:]
    #X_train = X_train[:-500]
    #y_train = y_train[:-500]



    # define the training set and labels
    X_val, y_val = X_train[-10000:], y_train[-10000:]


    print("Training set shape: ", X_train.shape)
    print("Validation set shape: ", X_val.shape)
    print("Test set shape: ", X_test.shape)
    # dictionary to store the k values as keys and the validation accuracies as the values
    val_accuracy_per_k = {}

    for k in [1,2,3,4,5,10,20]:
        print(f"Calculating validation accuracy for k={k}")
        val_accuracy_per_k[k] = knn_classifier(X_train, y_train, X_val, y_val, k, dist_func=euclidean_distance)
        print(f"Validation accuracy of {val_accuracy_per_k[k]} % for k={k}")

    best_k = max(val_accuracy_per_k, key=val_accuracy_per_k.get)


    print(f"Best validation accuracy of {val_accuracy_per_k[best_k]} % for k={best_k}")

    print("Running on the test set...")
    test_accuracy = knn_classifier(X_train, y_train, X_test, y_test, best_k, dist_func=euclidean_distance)
    print(test_accuracy)


    # Do the same for cosine distance
    val_accuracy_per_k = {}

    for k in [1,2,3,4,5,10,20]:
        print(f"Calculating validation accuracy for k={k}")
        val_accuracy_per_k[k] = knn_classifier(X_train, y_train, X_val, y_val, k, dist_func=cosine_distance)
        print(f"Validation accuracy of {val_accuracy_per_k[k]} % for k={k}")

    best_k = max(val_accuracy_per_k, key=val_accuracy_per_k.get)


    print(f"Best validation accuracy of {val_accuracy_per_k[best_k]} % for k={best_k}")

    print("Running on the test set...")
    test_accuracy = knn_classifier(X_train, y_train, X_test, y_test, best_k, dist_func=cosine_distance)
    print(test_accuracy)

