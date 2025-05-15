# This file defines the logic for serving requests to the user.
from flask import Flask
from .config import ServerConfig
import numpy as np
import faiss
import os
import struct
import json
from .api import init_api

# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize FAISS to do this
class Server:

    # obtain all the setup information from configuration
    def __init__(self, config: ServerConfig):
        self.config = config

        # directories
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        self.image_directory = config.image_directory
        self.index_type = config.index_type
        self.disk_index = config.disk_index

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

        # Initialize Flask app and API
        self.app = Flask(__name__)
        self.app.server = self
        self.api = init_api(self.app)
    
    def search(self, query):
        # Create random array embedding
        query_embedding = self.model.text_to_embeddings(query)
        if self.index_type == 'Memory':
            # Search for the k closest arrays
            D, I = self.faiss_index.search(query_embedding, self.k)

            search_results = []
            for i in range(I.shape[0]):
                for j in range(I.shape[1]):
                    # parse file information for page
                    pdf_name, _, page = self.npy_files[I[i][j]].rpartition('_')
                    page, _, _ = page.rpartition('.')
                    # create jpeg name
                    jpeg = self.image_directory + "/" + "/".join(pdf_name.rsplit("/", 2)[-2:]) + "_" + page + '.jpg'
                    
                    # add results to list
                    search_results.append({
                        "pdf": pdf_name, 
                        "page": page, 
                        "distance": float(D[i][j]), 
                        "jpeg": jpeg
                    })

        if self.index_type == 'Disk':
            normalized = query_embedding / np.linalg.norm(query_embedding)
            indices, distances = self.disk_index.search(normalized.flatten(), self.k, self.k * 2)
            search_results = []
            page_indices = os.path.join(self.embedding_directory, "page_indices.bin")
            with open(page_indices, "rb") as file:
                for i in range(len(indices)):
                    file.seek(indices[i] * 117, os.SEEK_SET)
                    pdf_name = file.read(113).decode('utf-8').strip()
                    page = str(struct.unpack("i", file.read(4))[0])
                    search_results.append({
                        "pdf": pdf_name,
                        "page": page,
                        "distance": float(distances[i])
                    })
        
        return {"results": search_results}

    def serve(self):
        # keep this function to maintain compatibility with scripts/start_server.py
        print("Welcome to End-Of-Term PDF Search Server")   
    
        print("Searching against " + str(self.faiss_index.ntotal) + " embeddings\n")
        try:
            while True:
                query = input("Search: ")
                # EOF detected
                if query == "":
                    continue

                result = self.search(query)
                json_object = json.dumps(result, indent=4)

                # print for testing
                print(json_object)



        except EOFError:
            print("\nThank you for using!")
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """Run the Flask server."""
        self.app.run(host=host, port=port, debug=debug)
