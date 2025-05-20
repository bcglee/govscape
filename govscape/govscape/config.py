# These classes should contain all the configuration information necessary for
# starting the server and serving queries, respectively.
import numpy as np
import os

class IndexConfig:
    def __init__(self, pdf_directory, embedding_directory, index_directory, image_directory, index_type):
        self.pdf_directory = pdf_directory
        self.embedding_directory = embedding_directory
        self.index_directory = index_directory
        self.image_directory = image_directory
        if index_type not in ["Memory", "Disk"]:
            raise ValueError("index_type must be either 'Memory' or 'Disk'")
        self.index_type = index_type
        self.dtype = np.float32

        
class ServerConfig:
    def __init__(self, index_config : IndexConfig, embedding_model, disk_index=None, k=3):
        self.pdf_directory = index_config.pdf_directory
        self.embedding_directory = index_config.embedding_directory
        self.index_directory = index_config.index_directory
        self.image_directory = index_config.image_directory
        self.model = embedding_model
        self.index_type = index_config.index_type
        self.disk_index = disk_index

        # define k for top-k
        self.k = k

        # define embedding size
        self.d = self.model.embedding_model.d

