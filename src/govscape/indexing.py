# This file contains the functionality for bulk loading and indexing the data 
# before requests can be served.
from config import IndexConfig
import diskannpy as dap
import os


class IndexBuilder:
    def __init__(self, config : IndexConfig):
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        self.dtype = config.dtype
        pass
    
    def build_index(self):
        # pass embedding directory to be transformed into single bin file (npy_to_bin)
        # config bin file every time? or append every time there's a new embedding?
        # return embedding.bin as embedding
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
            vector_dtype=self.dtype, 
            index_prefix="ann",  # ann is the default anyway. all files generated will have the prefix `ann_`, in the form of `f"{index_prefix}_"`
            pq_disk_bytes=0  # using product quantization of your vectors can still achieve excellent recall characteristics at a fraction of the latency, but we'll do it without PQ for now
        )


