#!/usr/bin/env python3

import os
import sys
import gc
from argparse import ArgumentParser, Namespace

import math
import cmath
import random
import time

from tqdm import tqdm, trange
from typing import List, Tuple, Dict, Set, Union, Optional, Literal, Any, Iterable, Callable, NamedTuple
from dataclasses import dataclass, field

import numpy as np
from numpy import array, ndarray, linalg, matrix, strings, testing

import matplotlib as mpl
from matplotlib import pyplot as plt, ticker

import numba
from numba.typed import List as NumbaList, Dict as NumbaDict
from numba import types, prange

from sim_hash import SimHash, test_simhash

# Load datasets
mtp_dataset: ndarray = np.load(os.path.expanduser("~/MTP/dataset.npy"), mmap_mode='r')
mtp_query_set: ndarray = np.load(os.path.expanduser("~/MTP/dataset_test.npy"), mmap_mode='r')
imagenet_dataset: ndarray = np.load(os.path.expanduser("~/tgtnn/data/imagenet/X.npy"), mmap_mode='r')
imagenet_query_set: ndarray = np.load(os.path.expanduser("~/tgtnn/data/imagenet/Q.npy"), mmap_mode='r')

# Initialize SimHash with parameters suitable for the datasets
hash_function = SimHash(
    num_hashes=500,
    num_bits=10,
    threshold=0,
    dimension=1000
)

# Compute hashes
mtp_hashes: ndarray = hash_function(mtp_dataset)
mtp_query_hashes: ndarray = hash_function(mtp_query_set)
imagenet_hashes: ndarray = hash_function(imagenet_dataset)
imagenet_query_hashes: ndarray = hash_function(imagenet_query_set)

# For MTP dataset: get histogram of number of matches across 100 queries
mtp_matches: ndarray = np.zeros((100, mtp_hashes.shape[0]), dtype=np.int32)
for qidx in tqdm(range(100), desc="Comparing MTP query hashes"):
    mtp_matches[qidx] = hash_function.compare_hashes(mtp_hashes, mtp_query_hashes[qidx]).astype(np.int32).sum(axis=1)

assert mtp_matches.dtype == np.int32
flattened_matches = mtp_matches.flatten()
hist_counts = np.bincount(flattened_matches, minlength=501)
print(hist_counts)

# For ImageNet dataset: get histogram of number of matches across 100 queries
imagenet_matches: ndarray = np.zeros((100, imagenet_hashes.shape[0]), dtype=np.int32)
for qidx in tqdm(range(100), desc="Comparing ImageNet query hashes"):
    imagenet_matches[qidx] = hash_function.compare_hashes(imagenet_hashes, imagenet_query_hashes[qidx]).astype(np.int32).sum(axis=1)

assert imagenet_matches.dtype == np.int32
flattened_matches = imagenet_matches.flatten()
hist_counts = np.bincount(flattened_matches, minlength=501)
print(hist_counts)