#!/usr/bin/env python3

import os
import sys
import numpy as np

NUM_QUERIES = 1000

def shard_maker(i : int) -> str:
    if (i < 10 and i >= 0):
        return "shard_0" + str(i)
    elif (i < 16 and i >= 10):
        return "shard_" + str(i)
    else:
        raise ValueError("Shard index out of range: {}".format(i))
    
# Write a function to run a process, wait till completion and return the output (cout)
def run_process(command: str) -> str:
    print("Running command: {}".format(command))
    result = os.popen(command).read().splitlines()
    return result

def run_shard(i: int) -> list:
    shard = shard_maker(i)
    X_csr = "../data/movielens/{}/X.csr".format(shard)
    Q_csr = "../data/movielens/{}/Q.csr".format(shard)
    command = "./optimized_gtnn {} {}".format(X_csr, Q_csr)
    output =  [line.split() for line in run_process(command)]
    return [float(output[0][3]), int(output[1][5]), float(output[2][3]), float(output[3][2]), float(output[4][2]), int(output[5][5])]

def final_values() -> list:
    times = [0.0] * 16
    num_dots = [0] * 16
    naive_times = [0.0] * 16
    precisions = [0.0] * 16
    recalls = [0.0] * 16
    num_vectors = [0] * 16
    for i in range(16):
        shard = run_shard(i)
        times[i] = shard[0]
        num_dots[i] = shard[1]
        naive_times[i] = shard[2]
        precisions[i] = shard[3]
        recalls[i] = shard[4]
        num_vectors[i] = shard[5] // NUM_QUERIES
    net_time = sum([times[i] * num_vectors[i] for i in range(16)]) / sum(num_vectors)
    net_num_dots = sum(num_dots)
    net_naive_time = sum([naive_times[i] * num_vectors[i] for i in range(16)]) / sum(num_vectors)
    net_precision = sum(precisions) / 16
    net_recall = sum(recalls) / 16
    net_num_vectors = sum(num_vectors)
    return [net_time, net_naive_time, net_precision, net_recall, net_num_dots, net_num_vectors]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "shard":
        shard_index = int(sys.argv[2])
        print(run_shard(shard_index))
    else:
        results = final_values()
        print("Net time: {:.3f} ms".format(results[0]))
        print("Net naive time: {:.3f} ms".format(results[1]))
        print("Net precision: {:.3f}".format(results[2]))
        print("Net recall: {:.3f}".format(results[3]))
        print("Total number of dot products: {}".format(results[4]))
        print("Total number of vectors processed: {}".format(results[5]))