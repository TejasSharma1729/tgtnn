#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import List
import sys, os, time
import argparse
import csv

import math, cmath, random, statistics
from tqdm import tqdm

import numpy as np
from numpy import array, ndarray, linalg, random as npr, matrix

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)
REPO_DIR = os.path.abspath(os.path.join(CUR_DIR, '..'))
DATA_DIR = os.path.join(REPO_DIR, 'data')

DATASETS: List[str] = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]

import gtnn
from gtnn import KNNSIndexDataset, ThresholdIndexDataset, read_matrix, save_matrix


@dataclass
class ResultRecord:
    """
    Stores the results of a nearest neighbor search test.
    """
    dataset: str
    is_knns: bool
    k_val: int = field(default=-1)
    threshold: float = field(default=0.0)
    double_group: bool = field(default=False)
    num_queries: int = field(default=0)
    avg_query_time_ms: float = field(default=0.0)
    avg_naive_time_ms: float = field(default=0.0)
    avg_recall: float = field(default=0.0)
    avg_precision: float = field(default=0.0)


def run_tests(
        dataset: str,
        is_knns: bool,
        k_val: int,
        threshold: float,
        double_group: bool,
        num_queries: int,
) -> ResultRecord:
    """
    Runs nearest neighbor search tests on the specified dataset, with the mode and parameters provided.
    
    :param dataset: The name of the dataset to use
    :type dataset: str
    :param is_knns: If set, runs k-NNS; otherwise, runs threshold-based search
    :type is_knns: bool
    :param k_val: The number of nearest neighbors to find (used if is_knns is True)
    :type k_val: int
    :param threshold: The similarity threshold for search (used if is_knns is False)
    :type threshold: float
    :param double_group: Whether to use the double group testing strategy
    :type double_group: bool
    :param num_queries: The number of queries to process (0 means all)
    :type num_queries: int
    :return: A record containing the results of the test
    :rtype: ResultRecord
    """
    dataset_dir: str = os.path.join(DATA_DIR, dataset)
    X = np.load(os.path.join(dataset_dir, "X.npy"))
    Q = np.load(os.path.join(dataset_dir, "Q.npy"))
    n, d = X.shape

    num_queries = min(Q.shape[0], num_queries) if num_queries > 0 else Q.shape[0]
    index: KNNSIndexDataset | ThresholdIndexDataset

    if is_knns:
        index = KNNSIndexDataset(X, k_val=k_val)
    else:
        index = ThresholdIndexDataset(X, threshold=threshold)
    
    result_record = ResultRecord(
        dataset=dataset,
        is_knns=is_knns,
        k_val=k_val,
        threshold=threshold,
        double_group=double_group,
        num_queries=num_queries,
    )

    query_results: List[List[int]] = []
    # Run the algorithm on queries
    if double_group:
        print("Using double group testing strategy for queries...")
        start_t = time.time()
        query_results, num_dots = index.search_multiple(Q[:num_queries])
        end_t = time.time()
        result_record.avg_query_time_ms = (end_t - start_t) * 1000 / num_queries
    else:
        print("Searching and verifying queries one by one...")
        net_query_time = 0.0
        for qidx in tqdm(range(num_queries), desc="Processing queries"):
            start_t = time.time()
            res, dots = index.search(Q[qidx])
            end_t = time.time()
            query_time = (end_t - start_t) * 1000
            net_query_time += query_time
            query_results.append(res)
        result_record.avg_query_time_ms = net_query_time / num_queries

    # Verification
    net_naive_time = 0.0
    net_recall = 0.0
    net_precision = 0.0
    print("Verifying results with naive search...")
    start_t = time.time()
    dots = Q[:num_queries] @ X.T
    for qidx in range(num_queries):
        if is_knns:
            true_neighbors = np.argsort(-dots[qidx])[:k_val]
        else:
            true_neighbors = np.where(dots[qidx] >= threshold)[0]
        true_neighbors = np.sort(true_neighbors)
        retrieved_neighbors = query_results[qidx]
        recall = len(set(true_neighbors) & set(retrieved_neighbors)) / len(true_neighbors) if len(true_neighbors) > 0 else 1.0
        precision = len(set(true_neighbors) & set(retrieved_neighbors)) / len(retrieved_neighbors) if len(retrieved_neighbors) > 0 else 1.0
        net_recall += recall
        net_precision += precision

        if (recall < 1.0) or (precision < 1.0):
            print(f"Query {qidx}: Recall={recall:.4f}, Precision={precision:.4f}, True Neighbors={true_neighbors}, Retrieved Neighbors={retrieved_neighbors}")
    end_t = time.time()
    result_record.avg_naive_time_ms = (end_t - start_t) * 1000 / num_queries
    
    result_record.avg_naive_time_ms = net_naive_time / num_queries
    result_record.avg_recall = net_recall / num_queries
    result_record.avg_precision = net_precision / num_queries

    return result_record


def save_results(results: List[ResultRecord], filename: str):
    """
    Saves the test results to a CSV file.
    
    :param results: The list of test results to save
    :type results: List[ResultRecord]
    :param filename: The name of the file to save the results to
    :type filename: str
    """
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dataset", "is_knns", "parameter", "double_group", "num_queries", "avg_query_time_ms", "avg_naive_time_ms", "avg_recall", "avg_precision"])
        for record in results:
            parameter = record.k_val if record.is_knns else record.threshold
            writer.writerow([
                record.dataset,
                record.is_knns,
                parameter,
                record.double_group,
                record.num_queries,
                record.avg_query_time_ms,
                record.avg_naive_time_ms,
                record.avg_recall,
                record.avg_precision,
            ])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GTNN Nearest Neighbor Search: Either k-NNS or threshold-based.")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=DATASETS + ["all"],
        default="all",
        help="Dataset to use for testing (default: all)",
    )
    parser.add_argument(
        "--problem",
        "-p",
        type=str,
        choices=["knns", "k", "threshold", "t", "all"],
        default="all",
        help="Mode of nearest neighbor search: 'knns' for k-NNS, 'threshold' for threshold-based (default: knns)",
    )
    parser.add_argument(
        "--k-val",
        "-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to search for",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Distance threshold for neighbors",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["binary_splitting", "b", "double_group_testing", "g", "all"],
        default="all",
        help="Use double group testing strategy if set",
    )
    parser.add_argument(
        "--num-queries",
        "-q",
        type=int,
        default=-1,
        help="Number of queries to test (default: all)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default="test_results.csv",
        help="Output CSV file to save results (default: test_results.csv)",
    )
    args = parser.parse_args()

    datasets_to_test: List[str] = DATASETS if args.dataset == "all" else [args.dataset]
    problems_to_test: List[str] = ["knns"] if args.problem in ["knns", "k"] \
                                    else ["threshold"] if args.problem in ["threshold", "t"]\
                                    else ["knns", "threshold"]
    modes_to_test: List[bool] = [False] if args.mode in ["binary_splitting", "b"] \
                                else [True] if args.mode in ["double_group_testing", "g"] \
                                else [False, True]
    
    results: List[ResultRecord] = []
    for dataset in datasets_to_test:
        for problem in problems_to_test:
            for double_group in modes_to_test:
                is_knns = (problem == "knns")
                result = run_tests(
                    dataset=dataset,
                    is_knns=is_knns,
                    k_val=args.k_val,
                    threshold=args.threshold,
                    double_group=double_group,
                    num_queries=args.num_queries,
                )
                results.append(result)
    save_results(results, args.output_file)