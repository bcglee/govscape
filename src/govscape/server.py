# This file defines the logic for serving requests to the user, and it assumes
# that the index has already been built.
from config import ServerConfig
from config import IndexConfig
import numpy as np
import faiss
import os

# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize FAISS to do this

def main():
    index_config = IndexConfig()
    server_config = ServerConfig(index_config, EmbeddingModel)

    # builds a preliminary index
    list_of_arrays = []
    index = {}
    directory = 'data/embeddings/'
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            array = np.load(file_path)
        index[len(list_of_arrays)] = file_path
        list_of_arrays.append(array)
    arrays = np.vstack(list_of_arrays)
    print(arrays)

    # array serving
    s = Server(server_config, arrays, index)
    s.serve(3)

class Server:
    # commented out to do searching with preliminary index
    # def __init__(self, config : ServerConfig):
    #     self.pdf_directory = config.pdf_directory
    #     self.embedding_directory = config.embedding_directory
    #     self.index_directory = config.index_directory
    #     pass

    def __init__(self, config : ServerConfig, arrays, index):
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = arrays
        self.index_directory = index
        pass

    # This function accepts a query to search and returns a List object of
    # file names that it believes serves the user's request
    def serve(self, k):
        print("Welcome to End-Of-Term PDF Search Server")

        # Creating Faiss model
        d = 5                           # dimension
        nq = 1                      # nb of queries
        faiss_index = faiss.IndexFlatL2(d)
        print(faiss_index.is_trained)
        faiss_index.add(self.embedding_directory)
        print("Searching against " + str(faiss_index.ntotal) + " embeddings\n")

        em = EmbeddingModel()
        try:
            while True:
                query = input("Search: ")
                #EOF detected
                if query == "":
                    break
                query_embedding = em.embed_query(query)
                D, I = faiss_index.search(query_embedding, k)
                print(f"This queries' embedding {query_embedding}\n")
                for i in range(I.shape[0]):
                    for j in range(I.shape[1]):
                         pdf_file = self.index_directory[I[i][j]]
                         print(f"{pdf_file} is at distance {D[i][j]}")
                print()

        except EOFError:
            print("\nThank you for using!")

# This class represents our embedding model, as a placeholder it returns
# a random vector
class EmbeddingModel:
    def __init__(self):
        pass

    def embed_query(self, query: str) -> np.array:
        random_embedding = np.array([np.random.rand(5)])
        return random_embedding

if __name__ == '__main__':
         main()