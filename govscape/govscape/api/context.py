class ServerContext:
    """Shared context for API resources"""
    
    def __init__(self, server):
        self.faiss_index = server.faiss_index
        self.model = server.model
        self.k = server.k
        self.npy_files = server.npy_files
        self.image_directory = server.image_directory
