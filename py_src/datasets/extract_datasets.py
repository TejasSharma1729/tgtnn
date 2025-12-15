#!/usr/bin/env python3

import os
import sys
from typing import List, Tuple, Dict, Set, Union, Optional, Any, Iterable, Callable, NamedTuple
from dataclasses import dataclass, field

import numpy as np
from numpy import array, ndarray, linalg, matrix, strings, testing
import numba

CURRENT_FILE: str = os.path.abspath(__file__)
CURRENT_DIR: str = os.path.dirname(CURRENT_FILE)
DATASET_BASE_PATH: str = "/mnt/Drive4/harsh/datasets"
DATASETS: List[str] = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]

def extract_dataset(dataset_dir: str, output_dir: Optional[str] = None) -> ndarray:
    """
    Extracts a dataset by name and saves it to the specified output directory.
    
    Args:
        dataset_name (str): The name of the dataset to extract.
        output_dir (str): The directory where the dataset will be saved.
    
    Returns:
        ndarray: The extracted dataset as a NumPy array.
    """
    # Get X.txt
    X_file_name = os.path.join(dataset_dir, "X.txt")
    X: ndarray
    with open(X_file_name, 'r') as f:
        X_data = f.readlines()
        X = array([[float(value) for value in line.strip().split(',')] for line in X_data])
    if output_dir:
        X_output_file = os.path.join(output_dir, "X.npy")
        np.save(X_output_file, X)
    
    # Get Q.txt
    Q_file_name = os.path.join(dataset_dir, "Q.txt")
    Q: ndarray
    with open(Q_file_name, 'r') as f:
        Q_data = f.readlines()
        Q = array([[float(value) for value in line.strip().split(',')] for line in Q_data])
    if output_dir:
        Q_output_file = os.path.join(output_dir, "Q.npy")
        np.save(Q_output_file, Q)


def get_dataset_and_output_directories(dataset_name: str) -> Tuple[str, str]:
    """
    Returns the dataset directory and output directory for a given dataset name.
    """
    dataset_parent: str = os.path.join(DATASET_BASE_PATH, dataset_name)
    dataset_dir: str = os.path.join(dataset_parent, "output")
    assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist."
    repo_root: str = os.path.join(CURRENT_DIR, os.pardir)
    output_parent: str = os.path.join(repo_root, "data")
    output_dir: str = os.path.join(output_parent, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return dataset_dir, output_dir


if __name__ == "__main__":
    for dataset_name in DATASETS:
        dataset_dir, output_dir = get_dataset_and_output_directories(dataset_name)
        print(f"Extracting dataset {dataset_name} from {dataset_dir} to {output_dir}")
        extract_dataset(dataset_dir, output_dir)
        print(f"Dataset {dataset_name} extracted successfully.")
    print("All datasets extracted successfully.")