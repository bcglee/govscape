# These classes should contain all the configuration information necessary for
# starting the server and serving queries, respectively.
import numpy as np

class IndexConfig:
    def __init__(self, pdf_directory, data_dir, index_type):
        self.pdf_directory = pdf_directory
        self.embedding_directory = data_dir + '/embeddings'
        self.embedding_img_pg_directory = data_dir + '/embeddings_img_pg'
        self.index_directory = data_dir + '/index'
        self.image_directory = data_dir + '/img'
        self.metadata_directory = data_dir + '/metadata'
        if index_type not in ["Memory", "Disk"]:
            raise ValueError("index_type must be either 'Memory' or 'Disk'")
        self.index_type = index_type
        self.dtype = np.float32

        
class ServerConfig:
    def __init__(self, index_config : IndexConfig, embedding_model, k=3):
        self.index_config= index_config
        self.pdf_directory = index_config.pdf_directory
        self.embedding_directory = index_config.embedding_directory
        self.embedding_img_pg_directory = index_config.embedding_img_pg_directory
        self.index_directory = index_config.index_directory
        self.image_directory = index_config.image_directory
        self.metadata_directory = index_config.metadata_directory
        self.model = embedding_model
        self.index_type = index_config.index_type

        # define k for top-k
        self.k = k

        # define embedding size
        self.d = self.model.d

