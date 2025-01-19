# This file defines the logic for serving requests to the user, and it assumes
# that the index has already been built.
from .config import ServerConfig

class Server:
    def __init__(self, config : ServerConfig):
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        pass

    # This function accepts a query to search and returns a List object of
    # file names that it believes serves the user's request
    def process(self, query):
        pass

    # This class represents our embedding model, as a placeholder it returns
    # a random vector
    class EmbeddingModel:
        def __init__(self):
            pass

        def embed_query(query -> string):
            




# basic pipeline developed:
# 1. accept a query
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query
# 4. utilize FAISS to do this