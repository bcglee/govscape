# This file contains the functionality for bulk loading and indexing the data 
# before requests can be served.
from .config import IndexConfig


class IndexBuilder:
    def __init__(self, config : IndexConfig):
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        pass



