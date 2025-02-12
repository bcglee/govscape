import numpy as np
import diskannpy as dap
from indexing import IndexBuilder
from config import IndexConfig
from sentence_transformers import SentenceTransformer
# bin_embeddings = "/homes/gws/alisony/govscape/src/govscape/data/bin_embeddings/embeddings.bin"

# # vecs = my_set_of_vectors / np.linalg.norm(my_set_of_vectors, axis=1)  # useful if your intention is to rank by a directionless 
# # cosine angle distance
# my_dtype = np.float32  # or np.uint8 or np.int8 ONLY
# # my_set_of_vectors: np.typing.NDArray[my_dtype] = bin_embeddings # your vectors come from somewhere - you need to bring these!
# # index_to_identifiers_map: np.typing.NDArray[str] = ... # your vectors likely have some kind of external identifier - 
# # you need to keep track of the external identifier -> index relationship somehow
# # identifiers_to_index_map: dict[str, np.uint32|np.uint.64] = ... # your map of your external id to the `diskannpy` internal id
# # diskannpy `query` responses will contain the _internal id only_, and if you don't have these maps you won't be able to 
# # know what this relates to


# dap.build_disk_index(
#     data=bin_embeddings,
#     distance_metric="l2", # can also be cosine, especially if you don't normalize your vectors like above
#     index_directory="/homes/gws/alisony/govscape/src/govscape/indices", 
#     complexity=128,  # the larger this is, the more candidate points we consider when ranking
#     graph_degree=64,  # the beauty of a vamana index is it's ability to shard and be able to transfer long distances across the grpah without navigating the whole thing. the larger this value is, the higher quality your results, but the longer it will take to build 
#     search_memory_maximum=16.0, # a floating point number to represent how much memory in GB we want to optimize for @ query time
#     build_memory_maximum=100.0, # a floating point number to represent how much memory in GB we are allocating for the index building process
#     num_threads=0,  # 0 means use all available threads - but if you are in a shared environment you may need to restrict how greedy you are
#     vector_dtype=my_dtype,  # we specified this in the Commonalities section above
#     index_prefix="ann",  # ann is the default anyway. all files generated will have the prefix `ann_`, in the form of `f"{index_prefix}_"`
#     pq_disk_bytes=0  # using product quantization of your vectors can still achieve excellent recall characteristics at a fraction of the latency, but we'll do it without PQ for now
# )

config = IndexConfig(
    pdf_directory="/homes/gws/alisony/govscape/src/govscape/data/pdfs",
    embedding_directory="/homes/gws/alisony/govscape/src/govscape/data/embeddings",
    index_directory="/homes/gws/alisony/govscape/src/govscape/indices"
)

index_builder = IndexBuilder(config)
index_builder.build_index()



# Load the same model used for indexing
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert "gold" to a query vector
query_text = "gold"
query_vector = model.encode(query_text).astype(np.float32)  # Ensure it's the same dtype
query_vector /= np.linalg.norm(query_vector)  # Normalize if needed

# some_index: np.uint32 = ... # the index in our `q` array of points that we will be using to query on an individual basis
# my_query_vector: np.typing.NDArray[my_dtype] = q[some_index] # make sure this is a 1-d array of the same dimensionality as your index!
# normalize if required by my_query_vector /= np.linalg.norm(my_query_vector)
index = dap.StaticDiskIndex(
    index_directory=config.index_directory,
    num_threads=0,
    num_nodes_to_cache=17,
    index_prefix="ann"  
)
internal_indices, distances = index.search(
    query=query_vector,
    k_neighbors=25,
    complexity=50,  # must be as big or bigger than `k_neighbors`
)
print(len(internal_indices))
page_indices = np.load("/homes/gws/alisony/govscape/src/govscape/data/embeddings/page_indices.npy")

# Map retrieved indices to page numbers
matching_pages = [page_indices[idx] for idx in internal_indices[0]]
print("Matching pages:", matching_pages)
