import sys
import os
import shutil
import numpy as np
import scipy.sparse as sp
import tempfile

# Add the big-ann-benchmarks repository to the Python path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(CUR_DIR, "..", "assets", "big-ann-benchmarks")
sys.path.append(BENCHMARK_PATH)

try:
    # Try to load libtbb.so.2 if needed for PISA
    import ctypes
    import os
    tbb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "big-ann-benchmarks", "tttt", "_skbuild", "libtbb", "libtbb.so.2")
    if os.path.exists(tbb_path):
        ctypes.CDLL(tbb_path, mode=ctypes.RTLD_GLOBAL)
    
    from neurips23.sparse.linscan.linscan import Linscan
    from neurips23.sparse.cufe.linscan import LinscanCUFE
    from neurips23.sparse.nle.nle import NLE
    from neurips23.sparse.shnsw.shnsw import SparseHNSW
    import neurips23.sparse.nle.interface as nle_interface
    # Disable PISA internal logging
    nle_interface.log_level(False)
    
    # Dense ANN libraries
    import faiss
    import scann
    import falconn
except ImportError as e:
    print(f"Warning: Could not import some benchmark modules. {e}")
except Exception as e:
    print(f"Warning: Error while loading benchmark libraries: {e}")

class BaselinePythonWrapper:
    """
    A wrapper to bridge numpy floating point datasets (as used in GTNN tests)
    to the original Python baseline implementations from big-ann-benchmarks.
    """
    def __init__(self, algo_instance, k=10):
        self.algo = algo_instance
        self.k = k

    def _to_sparse(self, mat):
        """
        Robustly converts dense or sparse input to CSR format.
        """
        if mat is None:
            return None
        if sp.issparse(mat):
            return mat.tocsr()
        return sp.csr_matrix(mat)

    def search_multiple(self, queries, use_threading=True):
        # Convert queries to sparse as expected by benchmark baselines
        sparse_queries = self._to_sparse(queries)
        
        # Ensure we have data to query
        if sparse_queries is None or sparse_queries.shape[0] == 0: # type: ignore
            return [], 0
            
        # Call the batch query method of the underlying algorithm
        # Note: Most original baselines don't support a threading flag here 
        # as they are already multi-threaded internally or purely single-threaded.
        self.algo.query(sparse_queries, self.k)
        
        # Retrieve results
        results = self.algo.get_results()
        
        all_indices = []
        if results is not None:
            for row in results:
                if len(row) > 0:
                    # Results can be a list of tuples (id, score) or a list of IDs.
                    # We always extract the ID (the first element of any tuple/list).
                    if isinstance(row[0], (tuple, list, np.ndarray)):
                        indices = [int(r[0]) for r in row]
                    else:
                        indices = [int(r) for r in row]
                    all_indices.append(indices)
                else:
                    all_indices.append([])
        
        # Ensure we return a list of length nq to match input queries
        nq = sparse_queries.shape[0] # type: ignore
        while len(all_indices) < nq:
            all_indices.append([])
            
        return all_indices, 0

    def streaming_update(self, new_data):
        print(f"Streaming update for {self.__class__.__name__} not implemented or supported natively.")
        pass

class LinscanWrapper(BaselinePythonWrapper):
    def __init__(self, data, k=10):
        try:
            from neurips23.sparse.linscan.linscan import Linscan
            algo = Linscan("ip", {})
            sparse_data: sp.csr_matrix = self._to_sparse(data) # type: ignore
            
            # Efficient iteration using CSR arrays directly.
            # Building dictionaries is how pylinscan consumes sparse data.
            indptr = sparse_data.indptr
            indices = sparse_data.indices
            data_vals = sparse_data.data
            
            print(f"Indexing {sparse_data.shape[0]} documents in Linscan...") # type: ignore
            for i in range(sparse_data.shape[0]): # type: ignore
                if i > 0 and i % 100000 == 0:
                    print(f"Linscan indexing progress: {i}/{sparse_data.shape[0]}...") # type: ignore
                start, end = indptr[i], indptr[i+1]
                # Convert to Python types to ensure compatibility with Rust/C++ bindings.
                row_dict = {int(indices[idx]): float(data_vals[idx]) for idx in range(start, end)}
                algo._index.insert(row_dict)
            
            print(f"Linscan Indexing complete: {algo._index}")
            super().__init__(algo, k)
        except Exception as e:
            print(f"Linscan init failed: {e}")
            raise

    def streaming_update(self, new_data):
        sparse_data: sp.csr_matrix = self._to_sparse(new_data) # type: ignore
        indptr = sparse_data.indptr
        indices = sparse_data.indices
        data_vals = sparse_data.data
        print(f"Updating Linscan with {sparse_data.shape[0]} documents...") # type: ignore
        for i in range(sparse_data.shape[0]): # type: ignore
            if i > 0 and i % 100000 == 0:
                print(f"Linscan update progress: {i}/{sparse_data.shape[0]}...") # type: ignore
            start, end = indptr[i], indptr[i+1]
            row_dict = {int(indices[idx]): float(data_vals[idx]) for idx in range(start, end)}
            self.algo._index.insert(row_dict)
        print(f"Linscan Update complete: {self.algo._index}")

class CufeWrapper(BaselinePythonWrapper):
    def __init__(self, data, k=10):
        try:
            from neurips23.sparse.cufe.linscan import LinscanCUFE
            algo = LinscanCUFE("ip", {})
            sparse_data: sp.csr_matrix = self._to_sparse(data) # type: ignore
            
            indptr = sparse_data.indptr
            indices = sparse_data.indices
            # Pre-quantize the entire data for speed
            q_data = np.round(sparse_data.data * algo.scale).astype(np.int32)
            
            print(f"Indexing {sparse_data.shape[0]} documents in CUFE...") # type: ignore
            for i in range(sparse_data.shape[0]): # type: ignore
                if i > 0 and i % 100000 == 0:
                    print(f"CUFE indexing progress: {i}/{sparse_data.shape[0]}...") # type: ignore
                start, end = indptr[i], indptr[i+1]
                row_dict = {int(indices[idx]): int(q_data[idx]) for idx in range(start, end)}
                algo._index.insert(row_dict)
            
            print(f"CUFE Indexing complete: {algo._index}")
            super().__init__(algo, k)
        except Exception as e:
            print(f"Cufe init failed: {e}")
            raise

    def streaming_update(self, new_data):
        sparse_data: sp.csr_matrix = self._to_sparse(new_data) # type: ignore
        indptr = sparse_data.indptr
        indices = sparse_data.indices
        q_data = np.round(sparse_data.data * self.algo.scale).astype(np.int32)
        print(f"Updating CUFE with {sparse_data.shape[0]} documents...") # type: ignore
        for i in range(sparse_data.shape[0]): # type: ignore
            if i > 0 and i % 100000 == 0:
                print(f"CUFE update progress: {i}/{sparse_data.shape[0]}...") # type: ignore
            start, end = indptr[i], indptr[i+1]
            row_dict = {int(indices[idx]): int(q_data[idx]) for idx in range(start, end)}
            self.algo._index.insert(row_dict)
        print(f"CUFE Update complete: {self.algo._index}")

class NLEWrapper(BaselinePythonWrapper):
    def __init__(self, data, k=10, t1=0, t2=500): # Set t1=0 for no pruning on base index
        self.temp_dir = tempfile.mkdtemp()
        try:
            from neurips23.sparse.nle.interface import PisaIndex
            # NLE uses two indices, base and full
            self.index_base = PisaIndex(os.path.join(self.temp_dir, "base"), overwrite=True)
            self.index_full = PisaIndex(os.path.join(self.temp_dir, "full"), overwrite=True)
            
            # Prepare data
            sparse_data: sp.csr_matrix = self._to_sparse(data) # type: ignore
            
            # Simple iterator for PisaIndex.index
            def data_iterator(batch_size=10000):
                for i in range(0, sparse_data.shape[0], batch_size): # type: ignore
                    yield sparse_data[i:i+batch_size]

            # Indexing
            # Suppress terminal spam during indexing
            with open(os.devnull, 'w') as fnull:
                old_stdout = sys.stdout
                sys.stdout = fnull
                try:
                    self.index_base.index(data_iterator(), t=t1)
                    self.index_full.index(data_iterator(), t=t2)
                    self.index_full.generate_forward()
                    self.index_base.load_inv(self.index_full, k=k, k1=-1, k2=-1, k3=100)
                    self.index_full.load_forward()
                finally:
                    sys.stdout = old_stdout
            
            # Create a mock NLE instance
            from neurips23.sparse.nle.nle import NLE
            algo = NLE("ip", {"t1": t1, "t2": t2})
            algo.index_base = self.index_base
            algo.index_full = self.index_full
            algo.k1 = -1
            algo.k2 = -1
            algo.k3 = 100
            
            super().__init__(algo, k)
        except Exception as e:
            print(f"NLE init failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def search_multiple(self, queries):
        sparse_queries = self._to_sparse(queries)
        # Suppress terminal prints
        with open(os.devnull, 'w') as fnull:
            old_stdout = sys.stdout
            sys.stdout = fnull
            try:
                self.algo.query(sparse_queries, self.k)
            finally:
                sys.stdout = old_stdout
        
        results = self.algo.get_results()
        
        # PISA results are often stored in self.algo.I as a numpy array
        all_indices = []
        if results is not None:
            # results is (nq, k) array of IDs
            for row in results:
                all_indices.append([int(idx) for idx in row])
        
        nq: int = sparse_queries.shape[0] # type: ignore
        while len(all_indices) < nq:
            all_indices.append([])
            
        return all_indices, 0

class FaissHNSWWrapper(BaselinePythonWrapper):
    def __init__(self, data, k=10, M=32, efConstruction=128, efSearch=64):
        # Convert to dense float32 for FAISS
        if sp.issparse(data):
            dense_data = data.toarray().astype(np.float32)
        else:
            dense_data = data.astype(np.float32)
            
        dim = dense_data.shape[1]
        # Using inner product metric as GTNN is based on it
        self._index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = efConstruction
        self._index.hnsw.efSearch = efSearch
        
        # Add data
        self._index.add(dense_data) # type: ignore
        super().__init__(self._index, k)

    def search_multiple(self, queries):
        if sp.issparse(queries):
            dense_queries = queries.toarray().astype(np.float32)
        else:
            dense_queries = queries.astype(np.float32)
            
        D, I = self.algo.search(dense_queries, self.k)
        return I.tolist(), 0

    def streaming_update(self, new_data):
        if sp.issparse(new_data):
            dense_data = new_data.toarray().astype(np.float32)
        else:
            dense_data = new_data.astype(np.float32)
        self.algo.add(dense_data)

class FaissGTWrapper(BaselinePythonWrapper):
    def __init__(self, data, k=10):
        if sp.issparse(data):
            dense_data = data.toarray().astype(np.float32)
        else:
            dense_data = data.astype(np.float32)
            
        dim = dense_data.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(dense_data) # type: ignore
        super().__init__(self._index, k)

    def search_multiple(self, queries):
        if sp.issparse(queries):
            dense_queries = queries.toarray().astype(np.float32)
        else:
            dense_queries = queries.astype(np.float32)
            
        D, I = self.algo.search(dense_queries, self.k)
        return I.tolist(), 0

    def streaming_update(self, new_data):
        if sp.issparse(new_data):
            dense_data = new_data.toarray().astype(np.float32)
        else:
            dense_data = new_data.astype(np.float32)
        self.algo.add(dense_data)

class ScannWrapper(BaselinePythonWrapper):
    def __init__(self, data, k=10, num_leaves=None):
        if sp.issparse(data):
            dense_data = data.toarray().astype(np.float32)
        else:
            dense_data = data.astype(np.float32)
            
        if num_leaves is None:
            num_leaves = int(np.sqrt(dense_data.shape[0]))
            
        self.searcher = scann.scann_ops_pybind.builder(dense_data, k, "dot_product").tree(
            num_leaves=num_leaves, 
            num_leaves_to_search=num_leaves//4, 
            training_sample_size=dense_data.shape[0]
        ).score_ah( # type: ignore
            2, anisotropic_quantization_threshold=0.2
        ).reorder(10 * k).build()
        super().__init__(self.searcher, k)

    def search_multiple(self, queries):
        if sp.issparse(queries):
            dense_queries = queries.toarray().astype(np.float32)
        else:
            dense_queries = queries.astype(np.float32)
            
        I, D = self.algo.search_batched(dense_queries, final_num_neighbors=self.k)
        return I.tolist(), 0

class FalconnWrapper(BaselinePythonWrapper):
    def __init__(self, data, k=10, num_tables=50):
        if sp.issparse(data):
            dense_data = data.toarray().astype(np.float32)
        else:
            dense_data = data.astype(np.float32)
            
        params = falconn.get_default_parameters(dense_data.shape[0], dense_data.shape[1], 
                                                distance=falconn.DistanceFunction.NegativeInnerProduct)
        # In newer FALCONN, 'l' is the number of hash tables
        params.l = num_tables
        # Internal FALCONN params setup
        # For small N like 100, 16 bits is way too many (2^16 > 100).
        # We need something like log2(N).
        hash_bits = int(np.log2(dense_data.shape[0])) if dense_data.shape[0] > 1 else 1
        falconn.compute_number_of_hash_functions(hash_bits, params)
        
        self.table = falconn.LSHIndex(params)
        self.table.setup(dense_data)
        self.query_object = self.table.construct_query_object()
        self.query_object.set_num_probes(num_tables * 2)
        
        super().__init__(self.query_object, k)

    def search_multiple(self, queries):
        if sp.issparse(queries):
            dense_queries = queries.toarray().astype(np.float32)
        else:
            dense_queries = queries.astype(np.float32)
            
        all_indices = []
        for q in dense_queries:
            indices = self.algo.find_k_nearest_neighbors(q, self.k)
            all_indices.append(list(indices))
        return all_indices, 0
    def __del__(self):
        try:
            import os
            import shutil
            if hasattr(self, "temp_dir") and self.temp_dir and os.path.exists(self.temp_dir): # type: ignore
                shutil.rmtree(self.temp_dir) # type: ignore
        except (ImportError, TypeError, AttributeError):
            pass

class SHNSWWrapper(BaselinePythonWrapper):
    def __init__(self, data, k=10, M=16, ef_construction=200):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_bin = os.path.join(self.temp_dir, "dataset.bin")
        try:
            from neurips23.sparse.shnsw.shnsw import SparseHNSW
            algo = SparseHNSW("ip", {"M": M, "efConstruction": ef_construction, "buildthreads": 16})
            import sparse_hnswlib
            
            # Prepare data
            sparse_data = sp.csr_matrix(data)
            
            # Write to binary format expected by SHNSW CSRMatrix
            with open(self.temp_bin, "wb") as f:
                f.write(np.int64(sparse_data.shape[0]).tobytes()) # type: ignore
                f.write(np.int64(sparse_data.shape[1]).tobytes()) # type: ignore
                f.write(np.int64(sparse_data.nnz).tobytes())
                f.write(sparse_data.indptr.astype(np.int64).tobytes())
                f.write(sparse_data.indices.astype(np.int32).tobytes())
                f.write(sparse_data.data.astype(np.float32).tobytes())

            p = sparse_hnswlib.Index(space="ip", dim=data.shape[1])
            p.init_index(max_elements=data.shape[0] + 1000, csr_path=self.temp_bin, ef_construction=ef_construction, M=M)
            p.add_items()
            algo.p = p
            algo.p.set_ef(10)
            super().__init__(algo, k)
        except Exception as e:
            print(f"SHNSW init failed: {e}")
            raise

    def __del__(self):
        try:
            import os
            import shutil
            if hasattr(self, "temp_dir") and self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except (ImportError, TypeError, AttributeError):
            pass

