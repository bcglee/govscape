# This file defines the logic for serving requests to the user, and it assumes
# that the index has already been built.
from config import ServerConfig
from config import IndexConfig
import numpy as np

# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize FAISS to do this

class Server:
    def __init__(self, config : ServerConfig):
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        pass

    # This function accepts a query to search and returns a List object of
    # file names that it believes serves the user's request
    def serve(self):
        print("Welcome to End-Of-Term PDF Search Server")
        em = EmbeddingModel()
        while True:
            query = input("Search:")
            #EOF detected
            if query == "":
                break
            query_embedding = em.embed_query(query)
            print(query_embedding)
        print("Thank you for using!")

# This class represents our embedding model, as a placeholder it returns
# a random vector
class EmbeddingModel:
    def __init__(self):
        pass

    def embed_query(self, query: str) -> np.array:
        random_embedding = np.random.rand(5)
        return random_embedding

def main():
    index_config = IndexConfig()
    server_config = ServerConfig(index_config, EmbeddingModel)
    s = Server(server_config)
    s.serve()

if __name__ == '__main__':
         main()