#!/usr/bin/env python3
import numpy as np
import sys

def generate_random_vector(dim : int, sparsity : float) -> tuple:
    values = []
    indices = []
    start = 0
    while True:
        shift = int(round(np.random.poisson(lam = 1 / sparsity, size = 1)[0]))
        if start + shift >= dim:
            break
        indices.append(start + shift)
        values.append(np.random.exponential(scale = 1, size = 1)[0])
        start += shift
    return (values, indices)

def generate_random_CSRlike_matrix(num : int, dim : int, sparsity: float) -> tuple:
    values = []
    indices = []
    indptrs = []
    ptr = 0
    for i in range(num):
        vector = generate_random_vector(dim, sparsity)
        values.extend(vector[0])
        indices.extend(vector[1])
        indptrs.append(ptr)
        ptr += len(vector[0])
    return (num, dim, ptr, values, indices, indptrs)

def save_CSRlike_matrix(matrix : tuple, filename : str):
    with open(filename, 'w') as f:
        f.write(str(matrix[0]) + ' ' + str(matrix[1]) + ' ' + str(matrix[2]) + '\n')
        for i in matrix[3]:
            f.write(str(i) + ' ')
        f.write('\n')
        for i in matrix[4]:
            f.write(str(i) + ' ')
        f.write('\n')
        for i in matrix[5]:
            f.write(str(i) + ' ')
        f.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: python3 temp.py <num> <query> <dim> <sparsity> <dataset>')
        sys.exit(1)
    matrix = generate_random_CSRlike_matrix(int(sys.argv[1]), int(sys.argv[3]), float(sys.argv[4]))
    query = generate_random_CSRlike_matrix(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
    print(matrix[0], matrix[1], matrix[2])
    print(query[0], query[1], query[2])
    save_CSRlike_matrix(matrix, sys.argv[5] + '/X.txt')
    save_CSRlike_matrix(query, sys.argv[5] + '/Q.txt')
