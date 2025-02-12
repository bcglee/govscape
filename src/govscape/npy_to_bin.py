import numpy as np
import os
from config import IndexConfig

# input dir of pdf embeddings as .npy files
npy_embeddings = config.embedding_directory
# npy_embeddings = "/homes/gws/alisony/govscape/src/govscape/data/embeddings"

bin_path = os.path.join(npy_embeddings, "embeddings.bin")

all_vectors = []
page_indices = []  # store indices of tokenized pages

# loop through subdirectories of embedding directory (pdfs)
for subdir in os.listdir(npy_embeddings):
    subdir_path = os.path.join(npy_embeddings, subdir)
    if os.path.isdir(subdir_path): 
        page_index = len(all_vectors)  # start index
        page_indices.append(page_index)  # save page index
        # loop through pages of pdf
        for page in os.listdir(subdir_path):
            if page.endswith(".npy"):
                npy_path = os.path.join(subdir_path, page)
                # load .npy file
                data = np.load(npy_path)
                all_vectors.append(data)

# list of arrays becomes single numpy array
all_vectors = np.concatenate(all_vectors, axis=0)

# save as .bin file with header
with open(bin_path, "wb") as f:
    np.array([all_vectors.shape[0], all_vectors.shape[1]], dtype=np.int32).tofile(f)  # Write header
    all_vectors.tofile(f)  # Write vector data

# Optional: Write page indices if you want to track page boundaries
# (You can use this for referencing later if needed, or include it in the same .bin file or separately)
page_indices_path = os.path.join(os.path.dirname(bin_path), "page_indices.npy")
np.save(page_indices_path, page_indices)

print(f"Combined .bin file saved to {bin_path}")
print(f"Page indices saved to {page_indices_path}")