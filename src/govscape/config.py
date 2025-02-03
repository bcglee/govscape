# These classes should contain all the configuration information necessary for
# starting the server and serving queries, respectively.
from pdf_to_embedding import PDFsToEmbeddings
import numpy as np
import faiss
import os

class IndexConfig:
    def __init__(self, pdf_directory = "test_data/pdfs", embedding_directory="test_data/embeddings", index_directory="test_data/index"):
        self.pdf_directory = pdf_directory
        self.embedding_directory = embedding_directory
        self.index_directory = index_directory

        
class ServerConfig:
    def __init__(self, index_config : IndexConfig, embedding_model, k=3, d = 512):
        self.pdf_directory = index_config.pdf_directory
        self.embedding_directory = index_config.embedding_directory
        self.index_directory = index_config.index_directory
        self.model = embedding_model
        self.model.pdfs_to_embeddings()

        # define k for top-k
        self.k = k

        # Creating Faiss model
        self.d = d

        self.faiss_index = faiss.IndexFlatL2(d)

        # Train model on test vectors
        self.npy_files = []
        for root, _, files in os.walk(self.embedding_directory):
            for file in files:
                if file.endswith('.npy'):  # Check for .npy extension
                    self.npy_files.append(os.path.join(root, file))
        
        self.arrays = [np.load(file) for file in self.npy_files]  # Load each .npy file into an array
        for array in self.arrays:
            print(array.shape)
        stacked_array = np.vstack(self.arrays) 
        self.faiss_index.add(stacked_array)