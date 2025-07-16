# This file contains the functionality for bulk loading and indexing the data 
# before requests can be served.
from .config import IndexConfig
import diskannpy as dap
import numpy as np
import os
from abc import ABC

class AbstractVectorIndex(ABC):
    def __init__(self, config : IndexConfig):
        pass
    
    def build_index(self):
        pass

    def load_index(self):
        pass    

    def search(self, query_vector, k):
        pass

class DiskANNIndex(AbstractVectorIndex):
    def __init__(self, config : IndexConfig):
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        self.dtype = config.dtype
        self.index = None
        self.page_indices = None
        pass

    def build_index(self):
        if not os.path.exists(self.index_directory):
            os.makedirs(self.index_directory)
        embedding = os.path.join(self.embedding_directory, "embeddings.bin")
        dap.build_disk_index( # comments from DiskANN repo
            data=embedding,
            distance_metric="l2", # can also be cosine, especially if you don't normalize your vectors like above
            index_directory=self.index_directory, 
            complexity=128,  # the larger this is, the more candidate points we consider when ranking
            graph_degree=64,  # the beauty of a vamana index is it's ability to shard and be able to transfer long distances across the grpah without navigating the whole thing. the larger this value is, the higher quality your results, but the longer it will take to build 
            search_memory_maximum=16.0, # a floating point number to represent how much memory in GB we want to optimize for @ query time
            build_memory_maximum=100.0, # a floating point number to represent how much memory in GB we are allocating for the index building process
            num_threads=0,  # 0 means use all available threads - but if you are in a shared environment you may need to restrict how greedy you are
            vector_dtype=np.float32, 
            index_prefix="ann",  # ann is the default anyway. all files generated will have the prefix `ann_`, in the form of `f"{index_prefix}_"`
            pq_disk_bytes=0  # using product quantization of your vectors can still achieve excellent recall characteristics at a fraction of the latency, but we'll do it without PQ for now
        )        

    def save_index(self, filepath):
        return

    def load_index(self):
        self.index = dap.StaticDiskIndex(
            index_directory=self.index_directory,
            num_threads=0,
            num_nodes_to_cache=17,
            index_prefix="ann"  
        )

    def search(self, query_vector, k, complexity):
        query_vector = query_vector / np.linalg.norm(query_vector)
        # query vectior should be 2D
        internal_indices, distances = self.index.search(
            query=query_vector,
            k_neighbors=k,
            complexity=complexity,  # must be as big or bigger than `k_neighbors`
        )
        return distances, internal_indices


class FAISSIndex(AbstractVectorIndex):
    def __init__(self, config : IndexConfig):
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        self.dtype = config.dtype
        self.index = None
        self.page_indices = None
        pass

    def build_index(self):
        # create a new index
        self.faiss_index = faiss.IndexFlatL2(self.d)

        # Train model on test vectors
        self.npy_files = []
        for root, _, files in os.walk(self.embedding_directory):
            for file in files:
                if file.endswith(".npy"):
                    self.npy_files.append(os.path.join(root, file))

        # Load each .npy file into an array
        self.arrays = [np.load(file) for file in self.npy_files]
        stacked_array = np.vstack(self.arrays)
        self.faiss_index.add(stacked_array)
        if not os.path.exists(self.index_directory):
            os.makedirs(self.index_directory)
        

    def save_index(self, filepath):
        return

    def load_index(self):
        return

    def search(self, query_vector, k, complexity):
        # query vectior should be 2D
        D, I = self.faiss_index.search(query_embedding, self.text_k)
        return D, I

