# This file defines the logic for serving requests to the user.
from flask import Flask, send_from_directory
from flask_cors import CORS
from .config import ServerConfig
import numpy as np
import sys
import os
import contextlib

# Avoid annoying output from faiss during import
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

with suppress_output():
    import faiss

import struct
import json
from .api import init_api
from .filter import Filter
from .indexing import DiskANNIndex, FAISSIndex

# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize FAISS to do this
class Server:
    # obtain all the setup information from configuration
    def __init__(self, config: ServerConfig):
        self.config = config

        # Directories
        self.pdf_directory = config.pdf_directory
        self.metadata_directory = config.metadata_directory
        self.embedding_directory = config.embedding_directory
        self.embedding_img_pg_directory = config.embedding_img_pg_directory
        self.index_directory = config.index_directory
        self.image_directory = config.image_directory
        self.index_type = config.index_type
        self.k = config.k
        self.k = config.k

        # Index configuration
        self.index_config = config.index_config

        # Model Params
        self.text_model = config.text_model
        self.text_d = config.text_d

        self.visual_model = config.visual_model
        self.visual_d = config.visual_d

        if self.index_type == 'Disk':
            self.index = DiskANNIndex(self.index_config)
            self.index.load_index()
        elif self.index_type == 'Memory':
            self.index = FAISSIndex(self.index_config)
            self.index.build_index()
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        self.filt = Filter(config)

        # Get the absolute path to the build directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.abspath(
            os.path.join(current_dir, "..", "..", "interface", "build")
        )

        print(f"Static files directory: {build_dir}")
        if not os.path.exists(build_dir):
            print(f"Warning: Build directory does not exist: {build_dir}")
            print("Please run 'npm run build' in the interface directory first")

        # Initialize Flask app and API
        self.app = Flask(__name__, static_folder=build_dir, static_url_path="")
        # TODO: Remove localhost:5173 soon for security concerns
        CORS(self.app, origins=["http://3.20.135.189", "http://localhost:5173"], supports_credentials=True)

        @self.app.route("/")
        def serve_index():
            return self.app.send_static_file("index.html")

        @self.app.route("/images/<path:filename>")
        def serve_image(filename):
            image_dir = os.path.abspath(self.image_directory)
            return send_from_directory(image_dir, filename)

        self.app.server = self
        self.api = init_api(self.app)

    def search(self, query, filters=None):
        query_embedding = self.text_model.encode_text(query)
    
        # Search for the k closest arrays
        D, pdf_names, pdf_pages = self.index.search(query_embedding, self.k)

        search_results = []
        for distance, name, page in zip(D, pdf_names, pdf_pages):
            jpeg_file = self.image_directory + "/" + name + "/" + name + "_" + page + '.jpeg'
            search_results.append({
                "pdf": pdf_name, 
                "page": page, 
                "distance": float(distance), 
                "jpeg": jpeg_file
            })
        
        if filters and search_results:
            search_results = self.filt.filter_results(search_results, filters)
            
        return {"results": search_results}

    def pdf_pages(self, pdf_id):
        """Get all page images for a PDF by pdf_id. Returns dict with 'images' key or error message."""
        if not pdf_id:
            return {"error": "Missing 'pdf_id' parameter"}, 400

        metadata_path = os.path.join(self.metadata_directory, pdf_id, "metadata.json")
        
        if not os.path.exists(metadata_path):
            return {"error": "Metadata not found"}, 404

        with open(metadata_path, "r") as f:
            meta = json.load(f)
        num_pages = meta.get("num_pages")
        if not num_pages:
            return {"error": "Page number not found"}, 404

        try:
            num_pages = int(num_pages)
        except Exception:
            raise Exception(f"Invalid page number in metadata: {metadata_path}")

        image_dir = os.path.join(self.image_directory, pdf_id)
        images = [f"{image_dir}/{pdf_id}_{i}.jpeg" for i in range(num_pages)]
        return {"images": images}

    def serve(self):
        # keep this function to maintain compatibility with scripts/start_server.py
        print("Welcome to End-Of-Term PDF Search Server")

        print("Searching against " + str(self.index.total_embeddings()) + " embeddings\n")
            while True:
                query = input("Search: ")
                # EOF detected
                if query == "":
                    continue

                result = self.search(query)
                json_object = json.dumps(result, indent=4)

                # print for testing
                print(json_object)

        except EOFError:
            print("\nThank you for using!")

    def run(self, host="0.0.0.0", port=8080, debug=False):
        """Run the Flask server."""
        self.app.run(host=host, port=port, debug=debug)
