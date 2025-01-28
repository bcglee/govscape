# This file defines the logic for serving requests to the user, and it assumes
# that the index has already been built.
from config import ServerConfig
from config import IndexConfig
import numpy as np
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
    server_config.model.pdfs_to_embeddings()
    # array serving
    s = Server(server_config)
    s.serve()

class Server:
    def __init__(self, config : ServerConfig):
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        self.model = config.model
        pass

    # Accepts a Query -> Prints out k closest arrays with distance
    def serve(self):
        print("Welcome to End-Of-Term PDF Search Server")

        # define k for top-k
        k = 3

        self.model.pdfs_to_embeddings()

        # Creating Faiss model
        d = 512  # dimension
        faiss_index = faiss.IndexFlatL2(d)
        print(faiss_index.is_trained)

        # Train model on test vectors
        npy_files = []
        for root, _, files in os.walk(self.embedding_directory):
            for file in files:
                if file.endswith('.npy'):  # Check for .npy extension
                    npy_files.append(os.path.join(root, file))
        
        arrays = [np.load(file) for file in npy_files]  # Load each .npy file into an array
        for array in arrays:
            print(array.shape)
        stacked_array = np.vstack(arrays) 
        faiss_index.add(stacked_array)
    
        print("Searching against " + str(faiss_index.ntotal) + " embeddings\n")
        try:
            while True:
                query = input("Search: ")
                # EOF detected
                if query == "":
                    continue
                # Create random array embedding
                query_embedding = self.model.text_to_embeddings(query)
                # Search for the three closest arrays
                D, I = faiss_index.search(query_embedding, k)
                print(f"This queries' embedding {query_embedding}\n")
                for i in range(I.shape[0]):
                    for j in range(I.shape[1]):
                         print(f"{npy_files[I[i][j]]} is at distance {D[i][j]}")
                print()
        except EOFError:
            print("\nThank you for using!")

if __name__ == '__main__':
         main()