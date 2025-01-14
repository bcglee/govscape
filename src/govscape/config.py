# These classes should contain all the configuration information necessary for
# starting the server and serving queries, respectively.

class IndexConfig:
    def __init__(self, pdf_directory = "data/pdfs", embedding_directory="data/embeddings", index_directory="data/index"):
        self.pdf_directory = pdf_directory
        self.embedding_directory = embedding_directory
        self.index_directory = index_directory

        
class ServerConfig:
    def __init__(self, index_config : IndexConfig, embedding_model):
        self.pdf_directory = index_config.pdf_directory
        self.embedding_directory = index_config.embedding_directory
        self.index_directory = index_config.index_directory
        self.model = embedding_model