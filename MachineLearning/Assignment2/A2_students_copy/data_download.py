import logging
import os

import zipfile
import requests
import tqdm
import sklearn as sk
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


source_urls = [
    "https://david-heurtel-depeiges.github.io/assets/data/MNIST_CSV.zip",
    "https://www.kaggle.com/api/v1/datasets/download/nelgiriyewithana/credit-card-fraud-detection-dataset-2023",
]
datasets = ["MNIST", "iris", "credit_card_fraud"]

zip_file_names = ["MNIST_CSV.zip", None, "credit-card-fraud-detection-dataset-2023"]


def download_file(url: str, output_path: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    with open(output_path, "wb") as file, tqdm.tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar:
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))
    LOGGER.info(f"Downloaded {url} to {output_path}")


def decompress_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    LOGGER.info(f"Extracted {zip_path} to {extract_to}")


def download_iris_dataset(save_path: str):
    iris = sk.datasets.load_iris()
    X = iris.data
    y = iris.target
    # Does a test-train split
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the datasets to CSV files, concatenating X and y
    train_data = np.column_stack((X_train, y_train))
    test_data = np.column_stack((X_test, y_test))
    np.savetxt(os.path.join(save_path, "iris_train.csv"), train_data, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(save_path, "iris_test.csv"), test_data, delimiter=",", fmt="%.6f")
    LOGGER.info(f"Saved iris dataset to {save_path}")


def main():
    # Download each file to the data directory ./data_cache
    data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(data_dir, exist_ok=True)

    for url in source_urls:
        file_name = os.path.basename(url)
        output_path = os.path.join(data_dir, file_name)
        download_file(url, output_path)
    LOGGER.info("All files downloaded.")

    # Decompress the zip files to /data/<dataset_name>
    for dataset, zip_file_name in zip(datasets, zip_file_names):
        extract_to = os.path.join(os.path.dirname(__file__), "data", dataset)
        os.makedirs(extract_to, exist_ok=True)
        if dataset == "iris":
            download_iris_dataset(extract_to)
        else:
            zip_file_path = os.path.join(data_dir, zip_file_name)
            try:
                decompress_zip(zip_file_path, extract_to)
            except:
                # Remove the .zip extension if it exists
                zip_file_path = zip_file_path.replace(".zip", "")
                decompress_zip(zip_file_path, extract_to)

    # For MNIST, the folders are MNIST/MNIST_CSV/...actual content
    # Move the content up one level
    mnist_dir = os.path.join(os.path.dirname(__file__), "data", "MNIST", "MNIST_CSV")
    if os.path.isdir(mnist_dir):
        for item in os.listdir(mnist_dir):
            s = os.path.join(mnist_dir, item)
            d = os.path.join(os.path.dirname(__file__), "data", "MNIST", item)
            if os.path.isdir(s):
                os.rename(s, d)
            else:
                os.replace(s, d)
        os.rmdir(mnist_dir)

        # For the creditcard fraud dataset, split the data into train and test sets
    credit_card_fraud_path = os.path.join(os.path.dirname(__file__), "data", "credit_card_fraud", "creditcard_2023.csv")
    if os.path.isfile(credit_card_fraud_path):
        df = pd.read_csv(credit_card_fraud_path)
        train_df, test_df = sk.model_selection.train_test_split(df, test_size=0.002, random_state=42)
        train_df.to_csv(
            os.path.join(os.path.dirname(__file__), "data", "credit_card_fraud", "credit_card_fraud_train.csv"),
            index=False,
        )
        test_df.to_csv(
            os.path.join(os.path.dirname(__file__), "data", "credit_card_fraud", "credit_card_fraud_test.csv"),
            index=False,
        )
        os.remove(credit_card_fraud_path)
        LOGGER.info("Split credit card fraud dataset into train and test sets.")

    # Verify the files.
    # For MNIST, we expect mnist_train.csv and mnist_test.csv
    expected_files = {
        "MNIST": ["mnist_train.csv", "mnist_test.csv"],
        "credit_card_fraud": ["credit_card_fraud.csv", "credit_card_fraud_test.csv"],
        "iris": ["iris_train.csv", "iris_test.csv"],
    }
    for dataset, files in expected_files.items():
        dataset_dir = os.path.join(os.path.dirname(__file__), "data", dataset)
        for file in files:
            file_path = os.path.join(dataset_dir, file)
            if not os.path.isfile(file_path):
                LOGGER.error(f"Expected file {file} not found in {dataset_dir}")
            else:
                LOGGER.info(f"Verified presence of {file} in {dataset_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
