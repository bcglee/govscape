# This file defines the logic for serving requests to the user.
from .config import ServerConfig
from .config import IndexConfig
import numpy as np
import json
import faiss
import os
from .pdf_to_embedding import PDFsToEmbeddings


# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize FAISS to do this
class Server:

    # obtain all the setup information from configuration
    def __init__(self, config : ServerConfig):
        self.config = config

        # directories
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        self.image_directory = config.image_directory

        # FAISS model
        self.model = config.model
        self.k = self.config.k
        self.d = self.config.d

        # create a new index
        self.faiss_index = faiss.IndexFlatL2(self.d)

        # Train model on test vectors
        self.npy_files = []
        for root, _, files in os.walk(self.embedding_directory):
            for file in files:
                if file.endswith('.npy'):
                    self.npy_files.append(os.path.join(root, file))

        # Load each .npy file into an array
        self.arrays = [np.load(file) for file in self.npy_files]
        stacked_array = np.vstack(self.arrays) 
        self.faiss_index.add(stacked_array)
        
    # Accepts a Query -> Returns JSON with closest results
    # Sample:
    # {
    #     "results": [
    #         {
    #             "pdf": "test_data/embeddings/gold/gold",
    #             "page": "0",
    #             "distance": 59.212852478027344
    #         },
    #         {
    #             "pdf": "test_data/embeddings/government/government",
    #             "page": "0",
    #             "distance": 68.0333251953125
    #         },
    #         {s
    #             "pdf": "test_data/embeddings/joebiden/joebiden",
    #             "page": "0",
    #             "distance": 68.0333251953125
    #         }
    #     ]
    # }

    def serve_test(self, file):
        print("Welcome to End-Of-Term PDF Search Server")   
    
        print("Searching against " + str(self.faiss_index.ntotal) + " embeddings\n")

        with open(file, 'r', encoding='utf-8') as f:
            pdf = f.readline()
            correct = 0
            while pdf:
                for _ in range(0, 10):
                    # Create random array embedding
                    search = f.readline()
                    print(search)
                    query_embedding = self.model.text_to_embeddings(search)
                    # Search for the k closest arrays
                    D, I = self.faiss_index.search(query_embedding, self.k)
                    search_results = []
                    for i in range(I.shape[0]):
                        for j in range(I.shape[1]):
                            # parse file information for page
                            pdf_name, _, page = self.npy_files[I[i][j]].rpartition('_')
                            print(pdf_name)
                            if pdf_name in pdf:
                                correct += 1
                print(pdf + " " + str(correct))
                pdf = f.readline()
