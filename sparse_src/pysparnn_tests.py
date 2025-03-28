#!/usr/bin/env python3
import numpy as np
from scipy.sparse import csr_matrix
from time import time
from pysparnn.cluster_index import ClusterIndex

def read_data(filename: str) -> csr_matrix:
    f = open(filename, "r")
    num_data, dimention, _ = f.readline().split()
    num_data = int(num_data)
    dimention = int(dimention)
    values = [float(s)  for s in f.readline().split()]
    indices = [int(s)  for s in f.readline().split()]
    indptrs = [int(s)  for s in f.readline().split()]
    data = csr_matrix((values, indices, indptrs), shape=(num_data, dimention))
    return csr_matrix(data)

def get_index(data: csr_matrix):
    labels = list('vec_' + str(i).zfill(6) for i in range(data.shape[0])) 
    index = ClusterIndex(data, labels)
    return index

def get_results(index : ClusterIndex, query: csr_matrix, k: int):
    return index.search(query, k)

if __name__ == "__main__":
    data = read_data("sparse_dataset/X.txt")
    query = read_data("sparse_dataset/Q.txt")
    index = get_index(data)
    start = time()
    results = get_results(index, query, 10)
    stop = time()
    print("Time per query: ", (stop - start) * 1000.0 / query.shape[0], "ms")
    np.savetxt("sparse_dataset/pysparnn.txt", results, fmt='%s')
    print("Results saved in sparse_dataset/pysparnn.txt")