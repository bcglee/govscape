# These classes should contain all the configuration information necessary for
# starting the server and serving queries, respectively.
import numpy as np
import os

class IndexConfig:
    def __init__(self, pdf_directory = "test_data/pdfs", embedding_directory="test_data/embeddings", index_directory="test_data/index", image_directory="test_data/images"):
        self.pdf_directory = pdf_directory
        self.embedding_directory = embedding_directory
        self.index_directory = index_directory
        self.image_directory = image_directory

        
class ServerConfig:
    def __init__(self, index_config : IndexConfig, embedding_model, k=3):
        self.pdf_directory = index_config.pdf_directory
        self.embedding_directory = index_config.embedding_directory
        self.index_directory = index_config.index_directory
        self.image_directory = index_config.image_directory
        self.model = embedding_model

        # define k for top-k
        self.k = k

        # define embedding size
        self.d = self.model.d

