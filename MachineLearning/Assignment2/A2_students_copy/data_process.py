import pandas as pd
import numpy as np


def preprocess_mnist_data(train_file_path: str, test_file_path: str):
    """Preprocess the MNIST data from CSV files.
    Args:
        train_file_path (str): Path to the training CSV file.
        test_file_path (str): Path to the testing CSV file.

    Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Preprocessed training and testing data, and their statistics.

    Expected types/shapes:
        - X_train: (60000, 784), dtype=np.float32
        - y_train: (60000,)
        - X_test: (10000, 784), dtype=np.float32
        - y_test: (10000,)
        - mean: (784,), dtype=np.float32
        - std: (784,), dtype=np.float32
    """

    train_data = pd.read_csv(train_file_path, header=None)
    test_data = pd.read_csv(test_file_path, header=None)

    X_train = train_data.iloc[:, 1:].values.astype(np.float32)
    y_train = train_data.iloc[:, 0].values

    X_test = test_data.iloc[:, 1:].values.astype(np.float32)
    y_test = test_data.iloc[:, 0].values

    mean = X_train.mean()
    std = X_train.std()

    # éviter division par zéro
    if std < 1e-6:
        std = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    mean = np.float32(mean)
    std = np.float32(std)

    return X_train, y_train, X_test, y_test, mean, std


def preprocess_credit_card(train_file_path: str, test_file_path: str):
    """
        This function should be very similar to the preprocess mnist data function except that we
        have a header, the classes column has name Class and you can discard the column:
    • Open the pair of csv files as input using pandas, keeping the header and using
    column id as our index
    • extract the column Class for both,
    • perform scaling column wise: substract, for each column, its mean and divide
    by its standard deviation, as computed on the train set.
    • return the result as numpy arrays
        Args:
            train_file_path (str): Path to the training CSV file.
            test_file_path (str): Path to the testing CSV file.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]: Preprocessed training and testing data, and their statistics.
        Expected types/shapes:
            - X_train: (n_samples_train, n_features), dtype=np.float32
            - y_train: (n_samples_train,)
            - X_test: (n_samples_test, n_features), dtype=np.float32
            - y_test: (n_samples_test,)
            - mean: (n_features,), dtype=np.float32
            - std: (n_features,), dtype=np.float32
    """

    train_data = pd.read_csv(train_file_path, index_col="id" )
    test_data = pd.read_csv(test_file_path, index_col="id")

    y_train = train_data["Class"].values
    y_test = test_data["Class"].values

    X_train = train_data.drop(columns=["Class"]).values.astype(np.float32)
    X_test = test_data.drop(columns=["Class"]).values.astype(np.float32)

    mean = X_train.mean(axis = 0)
    std = X_train.std(axis = 0)

    #evite la division par l'array zero
    std = np.where(std < 1e-6, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std


    return X_train, y_train, X_test, y_test, mean, std
