from dataset_utils import read_binary_csr, get_dataset, calculate_recall
from mlgt_sparse import BloomHashFunction, DenseSRPHasher, MinHasher, SparseSRPHasher

import os
CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
# Now test if it works -- I added the correct path to the X.csr AND the reading method.

X = read_binary_csr(os.path.join(CUR_DIR, "..", "data", "sparse1M", "X.csr"))
hasher = MinHasher() # type: ignore -- This is a class, the pyright stupidly sees a module.
res = hasher(X[0].data, X[0].indices, X[0].nnz)
print("Success! Hash value:", res)