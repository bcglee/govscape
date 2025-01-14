# This file defines the logic for serving requests to the user, and it assumes
# that the index has already been built.
from .config import ServerConfig

class Server:
    def __init__(self, config : ServerConfig):
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        pass
