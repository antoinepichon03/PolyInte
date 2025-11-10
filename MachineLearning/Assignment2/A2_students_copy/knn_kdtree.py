import numpy as np


class KDNode:
    """A node in a k-d tree."""

    def __init__(self, data, label, dim, left=None, right=None):
        """
        data: The feature vector of the point at this node.
        label: The class label of the point.
        dim: The dimension used for splitting at this node.
        left: The left child node.
        right: The right child node.
        """
        self.data = data
        self.label = label
        self.dim = dim
        self.left = left
        self.right = right


class KDTree:
    """A k-d tree data structure for efficient nearest neighbor search."""

    def __init__(self, X_train, y_train):
        """
        X_train: The training features.
        y_train: The training labels.

        Attributes:
            - dims: Number of dimensions in the feature space (will be needed for building the tree)
            - root: The root node of the k-d tree
        """
        self.dims = X_train.shape[1]
        self.root = self._build_tree(X_train, y_train, depth=0)

    def _build_tree(self, X_train, y_train, depth):
        """
        Recursively builds the k-d tree.
        X_train: The subset of training features for this node.
        y_train: The corresponding labels.
        depth: The current recursion depth.
        """
        if len(X_train) == 0:
            return None

        #Point sans enfant
        if len(X_train) == 1:
            return KDNode(data=X_train[0], label=y_train[0], dim=None, left=None, right=None)
        
        #Construction de manière récursive
        dimention = depth % self.dims

        #Tri des données selon la dimension courante
        sorted_indices = X_train[:, dimention].argsort()
        X_sorted = X_train[sorted_indices]
        y_sorted = y_train[sorted_indices]


        indice_median = len(X_sorted)//2
        median_point = X_sorted[indice_median]
        median_label = y_sorted[indice_median]

        X_left = X_sorted[:indice_median]
        y_left = y_sorted[:indice_median]
        X_right = X_sorted[indice_median + 1:]
        y_right = y_sorted[indice_median + 1:]

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return KDNode(data=median_point, label=median_label, dim=dimention, left=left_child, right=right_child)


    def _find_nearest(self, node, query_point, best_guess=None, best_dist=np.inf):
        """
        Recursively finds the nearest neighbor to a query point.
        node: The current node in the tree.
        query_point: The point for which to find the nearest neighbor.
        best_guess: The current best neighbor found so far.
        best_dist: The distance to the current best guess.

        Returns:
            best_guess: The nearest neighbor found.
            best_dist: The distance to the nearest neighbor.
        """
        if node is None:
            return best_guess, best_dist

        # Calcul de la distance 
        current_dist = np.linalg.norm(query_point - node.data)

        if current_dist < best_dist:
            best_guess = node
            best_dist = current_dist
        
        #si c'est une feuille, on a parcouru tout l'arbre, on peut donc retourner le resultat
        if node.left is None and node.right is None:
            return best_guess, best_dist

        if query_point[node.dim] < node.data[node.dim]:
            prio = node.left 
            secc = node.right
        else:
            prio = node.right
            secc = node.left
        
        #On explore d'abord la branche prioritaire
        best_guess, best_dist = self._find_nearest(prio, query_point, best_guess, best_dist)
        dist = abs(query_point[node.dim] - node.data[node.dim])
        if dist < best_dist:
            best_guess, best_dist = self._find_nearest(secc, query_point, best_guess, best_dist)

        return best_guess, best_dist

    def find_nearest_neighbor(self, query_point):
        """
        Public method to find the nearest neighbor.
        query_point: The point for which to find the nearest neighbor.
        """
        return self._find_nearest(self.root, query_point)[0]


def kdtree_1nn_classifier(X_train, y_train, X_test):
    """
    Classifies a set of test points using a 1-NN search with a k-d tree.
    X_train: The training features. np.ndarray of shape (num_train_samples, num_features)
    y_train: The training labels. np.ndarray of shape (num_train_samples,)
    X_test: The test features. np.ndarray of shape (num_test_samples, num_features)

    Returns:
        predictions: The predicted labels for the test set. np.ndarray of shape (num_test_samples,)
    """
    predictions = []
    tree = KDTree(X_train, y_train)

    for test_point in X_test:
        
        nearest_label = tree.find_nearest_neighbor(test_point).label
        predictions.append(nearest_label)

    return np.array(predictions)


if __name__ == "__main__":
    # Example usage
    from data_process import preprocess_mnist_data, preprocess_credit_card

    # Load and preprocess the MNIST dataset
    # X_train, y_train, X_test, y_test, mean, std = preprocess_mnist_data("data/MNIST/train.csv", "data/MNIST/t10k.csv")
    X_train, y_train, X_test, y_test, mean, std = preprocess_credit_card(
        "data/credit_card_fraud/credit_card_fraud_train.csv", "data/credit_card_fraud/credit_card_fraud_test.csv"
    )
    print("Data loaded and preprocessed.")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    # Use a subset for quick testing (you MUST remove this in real use)
    #X_train_small = X_train[:1000]
    #y_train_small = y_train[:1000]
    #X_test_small = X_test[:100]
    #y_test_small = y_test[:100]

    # Classify using k-d tree 1-NN
    predictions = kdtree_1nn_classifier(X_train, y_train, X_test)

    # Print results
    print("Predictions:", predictions)
    print("True labels:", y_test)

    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
