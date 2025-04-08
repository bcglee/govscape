from flask import Flask
from .config import ServerConfig
import numpy as np
import faiss
import os
from .api import init_api
from .api.context import ServerContext

class Server:
    """A Flask-based RESTful API for the GovScape search engine."""
    
    def __init__(self, config: ServerConfig):
        """Initialize the API with a Flask app and configuration."""
        self.app = Flask(__name__)
        self.config = config
        
        # directories
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        self.image_directory = config.image_directory
        
        # FAISS model
        self.model = config.model
        self.k = self.config.k
        self.d = self.config.d
        
        # create a new index
        self.faiss_index = faiss.IndexFlatL2(self.d)
        
        # Train model on test vectors
        self.npy_files = []
        for root, _, files in os.walk(self.embedding_directory):
            for file in files:
                if file.endswith('.npy'):
                    self.npy_files.append(os.path.join(root, file))
        
        # Load each .npy file into an array
        self.arrays = [np.load(file) for file in self.npy_files]
        if self.arrays:
            stacked_array = np.vstack(self.arrays)
            self.faiss_index.add(stacked_array)
        
        # Create shared context
        self.context = ServerContext(self)
        
        # Initialize API
        self.api = init_api(self.app)
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """Run the Flask server."""
        # Make context available to Resource classes
        self.app.context = self.context
        self.app.run(host=host, port=port, debug=debug)
