# This file defines the logic for serving requests to the user, and it assumes
# that the index has already been built.
from config import ServerConfig
from config import IndexConfig
import numpy as np
import json
import faiss
import os
from pdf_to_embedding import PDFsToEmbeddings

# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize FAISS to do this

def main():
    index_config = IndexConfig()
    server_config = ServerConfig(index_config, PDFsToEmbeddings('test_data/pdfs', 'test_data/text', 'test_data/embeddings'))
    s = Server(server_config)
    s.serve()

class Server:
    def __init__(self, config : ServerConfig):
        self.config = config
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        self.model = config.model
        self.k = self.config.k
        self.d = self.config.d
        self.faiss_index = self.config.faiss_index
        self.npy_files = self.config.npy_files

    # Accepts a Query -> Prints out k closest arrays with distance
    def serve(self):
        print("Welcome to End-Of-Term PDF Search Server")   
    
        print("Searching against " + str(self.faiss_index.ntotal) + " embeddings\n")
        try:
            while True:
                query = input("Search: ")
                # EOF detected
                if query == "":
                    continue
                # Create random array embedding
                query_embedding = self.model.text_to_embeddings(query)
                # Search for the three closest arrays
                D, I = self.faiss_index.search(query_embedding, self.k)

                search_results = []
                for i in range(I.shape[0]):
                    for j in range(I.shape[1]):
                        pdf_name, _, page = self.npy_files[I[i][j]].rpartition('_')
                        page, _, _ = page.rpartition('.')
                        search_results.append({"pdf": pdf_name, "page": page, "distance": float(D[i][j])})
                json_object = json.dumps({"results": search_results}, indent=4)
                print(json_object)
        except EOFError:
            print("\nThank you for using!")

if __name__ == '__main__':
         main()