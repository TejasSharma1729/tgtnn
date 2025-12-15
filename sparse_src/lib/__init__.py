"""
Sparse GTnn - High-Performance Sparse Vector Search Library

This module provides Python bindings for optimized C++ implementations of
sparse vector similarity search algorithms.

Available Modules (C++17, Compiled to .so):
    - sparse_types: Core sparse vector/matrix type definitions
    - sparse_csr_create: Compressed sparse row creation utilities
    - sparse_csr_read: CSR matrix reading utilities
    - knn_index_dataset: K-nearest neighbor search with dataset indexing
    - knn_index_double_group: Dual-level hierarchical K-NN indexing
    - threshold_index_dataset: Threshold-based search with dataset indexing
    - threshold_index_double_group: Dual-level hierarchical threshold search
    - threshold_index_randomized: Randomized partitioning for threshold search
    - threshold_index_sparse_optimized: Group testing with Eigen sparse backend (advanced)
    - threshold_linscan_naive: Baseline & LINSCAN comparison implementations
    - threshold_linscan_optimized: Optimized LINSCAN with inverted index batching

Features:
    - PascalCase class names for Python compatibility
    - Full std:: qualified modern C++17 code
    - NumPy-compatible interfaces via pybind11
    - Multi-threaded search operations (NUM_THREADS=16)
    - Inverted index optimization
    - Binary/Quad tree hierarchical search
    - Group testing with recursive bisection

Compilation:
    All 11 modules compile with C++17 standard and pybind11.
    Binaries in lib/*.so are ready for import.
"""

import os
import sys
from pathlib import Path
from glob import glob
from importlib.util import spec_from_file_location, module_from_spec
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"
__author__ = "Tejas Sharma"

_lib_dir = Path(__file__).parent
_status_file = _lib_dir / "COMPILATION_STATUS.txt"

# Load compiled .so modules if they exist
_so_modules: dict[str, Any] = {}
for so_file in glob(str(_lib_dir / "*.so")):
    mod_name = Path(so_file).stem
    try:
        spec = spec_from_file_location(mod_name, so_file)
        if spec and spec.loader:
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            _so_modules[mod_name] = mod
            globals()[mod_name] = mod
    except Exception as e:
        print(f"Warning: Failed to load {mod_name}: {e}", file=sys.stderr)

# Export compiled modules
__all__ = [
    "sparse_types",
    "sparse_csr_create",
    "sparse_csr_read",
    "knn_index_dataset",
    "knn_index_double_group",
    "threshold_index_dataset",
    "threshold_index_double_group",
    "threshold_index_randomized",
    "threshold_index_sparse_optimized",
    "threshold_linscan_naive",
    "threshold_linscan_optimized",
]

# Stub declarations for type checking (suppress linter warnings for .so modules)
sparse_types: Any
sparse_csr_create: Any
sparse_csr_read: Any
knn_index_dataset: Any
knn_index_double_group: Any
threshold_index_dataset: Any
threshold_index_double_group: Any
threshold_index_randomized: Any
threshold_index_sparse_optimized: Any
threshold_linscan_naive: Any
threshold_linscan_optimized: Any

# Module descriptions
MODULE_DESCRIPTIONS = {
    "sparse_types": "Core sparse vector/matrix types: SparseElem, SparseVec, SparseMat, dot_product(), add_sparse()",
    "sparse_csr_create": "CSR matrix creation utilities from dense/triplet formats",
    "sparse_csr_read": "Binary CSR matrix reading utilities",
    "knn_index_dataset": "KNNIndexDataset<N,K> - Binary tree K-NN with dataset-level pooling",
    "knn_index_double_group": "KNNIndexDoubleGroup - Quad-tree dual-level hierarchical K-NN",
    "threshold_index_dataset": "ThresholdIndexDataset<N> - Threshold search with binary tree indexing",
    "threshold_index_double_group": "ThresholdIndexDoubleGroup - Quad-tree threshold search (data × query)",
    "threshold_index_randomized": "ThresholdIndexRandomized<N,Q> - Stochastic partitioning for threshold search",
    "threshold_index_sparse_optimized": "ThresholdIndexSparseOptimized<N,Q> - Group testing with Eigen (most advanced)",
    "threshold_linscan_naive": "ThresholdNaiveSearch, ThresholdLinscanStyle, ThresholdLinscanOptimized baselines",
    "threshold_linscan_optimized": "ThresholdLinscanOptimized<S>, ThresholdBatchProcessor<S> - Inverted index + batch processing",
}

def get_module_info(module_name):
    """Get information about a compiled module."""
    return MODULE_DESCRIPTIONS.get(module_name, "No description available")

def list_modules():
    """List all available modules with descriptions."""
    print("\nAvailable Sparse GTnn Modules:")
    print("=" * 70)
    for mod_name, desc in MODULE_DESCRIPTIONS.items():
        status = "✓ Compiled" if mod_name in _so_modules else "⚠ Not compiled"
        print(f"\n{mod_name} [{status}]")
        print(f"  {desc}")
    print("=" * 70)

def print_status():
    """Print compilation status."""
    if _status_file.exists():
        print(_status_file.read_text())
    else:
        print("No compilation status file found.")
        print("Run 'python3 build_headers.py' to check compilation status.")

