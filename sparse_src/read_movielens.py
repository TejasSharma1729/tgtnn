#!/usr/bin/env python3
import numpy as np
import sys

def array_to_matrix(arr : np.ndarray) -> list:
    matrix = []
    user = []
    for i in range(arr.shape[0]):
        if (i > 0 and arr[i][0] != arr[i-1][0]):
            matrix.append(user)
            user = []
        user.append(arr[i][1])
    matrix.append(user)
    return matrix

def read_dataset_userwise() -> list:
    dset_files = np.load("ml-20mx16x32/trainx16x32_" + str(int(sys.argv[1])) + ".npz")
    arr = dset_files['arr_0']
    return array_to_matrix(arr)

def get_random_queries(dataset : list, num_queries : int) -> list:
    query_indices = np.random.choice(len(dataset), num_queries, replace=False)
    query_indices.sort()
    query_set = []
    for i in query_indices:
        query_set.append(dataset[i])
    return query_set

def write_dataset(dataset : list, file : str):
    # 0, 1 matrix: NO. values -- all 1s.
    num_nonzero = sum([len(user) for user in dataset])
    num_vectors = len(dataset)
    dimention = max(855776, max([max(user) for user in dataset]) + 1)
    writer = open(file, "w")
    writer.write(str(num_vectors) + " " + str(dimention) + " " + str(num_nonzero) + "\n")
    for _ in range (num_nonzero):
        writer.write("1 ")
    writer.write("\n")
    for user in dataset:
        for movie in user:
            writer.write(str(movie) + " ")
    writer.write("\n")
    indptr = 0
    for user in dataset:
        indptr += len(user)
        writer.write(str(indptr - len(user)) + " ")
    writer.write("\n")
    writer.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 read_movielens.py <dataset_number>")
        sys.exit(1)
    dataset = read_dataset_userwise()
    queries = get_random_queries(dataset, 1000)
    write_dataset(queries, "movielens/shard_" + sys.argv[1] + "/Q.txt")
    write_dataset(dataset, "movielens/shard_" + sys.argv[1] + "/X.txt")