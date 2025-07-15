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

import os
import struct
import json
from .api import init_api
from .filter import Filter
from .indexing import IndexBuilder

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

        # Model Params
        self.model = config.model
        self.k = config.k
        self.d = config.d

        if self.index_type == 'Disk':
            disk_index = IndexBuilder(self.index_config)
            disk_index.load_index()
            self.disk_index = disk_index

        elif self.index_type == 'Memory':
            # create a new index
            self.faiss_index = faiss.IndexFlatL2(self.d)

            # Train model on test vectors
            self.npy_files = []
            for root, _, files in os.walk(self.embedding_directory):
                for file in files:
                    if file.endswith(".npy"):
                        self.npy_files.append(os.path.join(root, file))

            # Load each .npy file into an array
            self.arrays = [np.load(file) for file in self.npy_files]
            stacked_array = np.vstack(self.arrays)
            self.faiss_index.add(stacked_array)

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
        query_embedding = self.model.encode_text(query)
        query_embedding = query_embedding[np.newaxis, :]
        search_results = []
        
        if self.index_type == 'Memory':
            # Search for the k closest arrays
            D, I = self.faiss_index.search(query_embedding, self.k)

            for i in range(I.shape[0]):
                for j in range(I.shape[1]):
                    # parse file information for page
                    pdf_name, _, page = self.npy_files[I[i][j]].rpartition('_')
                    page, _, _ = page.rpartition('.')
                    # create jpeg name
                    jpeg = self.image_directory + "/" + "/".join(pdf_name.rsplit("/", 2)[-2:]) + "_" + page + '.jpeg'
                    
                    # add results to list
                    search_results.append({
                        "pdf": pdf_name, 
                        "page": page, 
                        "distance": float(D[i][j]), 
                        "jpeg": jpeg
                    })

        elif self.index_type == 'Disk':
            normalized = query_embedding / np.linalg.norm(query_embedding)
            indices, distances = self.disk_index.search(normalized.flatten(), self.k, self.k * 2)
            page_indices = os.path.join(self.embedding_directory, "page_indices.bin")
            with open(page_indices, "rb") as file:
                for i in range(len(indices)):
                    file.seek(indices[i] * 117, os.SEEK_SET)
                    pdf_name = file.read(113).decode('utf-8').strip()
                    page = str(struct.unpack("i", file.read(4))[0])
                    jpeg = self.image_directory + "/" + "/".join(pdf_name.rsplit("/", 2)[-2:]) + "_" + page + '.jpeg'

                    search_results.append({
                        "pdf": pdf_name,
                        "page": page,
                        "distance": float(distances[i]),
                        "jpeg": jpeg
                    })
        
        if filters and search_results:
            search_results = self.filt.filter_results(search_results, filters)
            
        return {"results": search_results}

    def pdf_pages(self, pdf_id):
        """Get all page images for a PDF by pdf_id. Returns dict with 'images' key or error message."""
        if not pdf_id:
            return {"error": "Missing 'pdf_id' parameter"}, 400

        embedding_dir = os.path.join(self.embedding_directory, pdf_id)
        metadata_path = os.path.join(embedding_dir, f"{pdf_id}.json")

        if not os.path.exists(metadata_path):
            return {"error": "Metadata not found"}, 404

        with open(metadata_path, "r") as f:
            meta = json.load(f)
        page_nums = meta.get("page_nums")
        if not page_nums:
            return {"error": "Page number not found"}, 404

        try:
            page_nums = int(page_nums)
        except Exception:
            return {"error": "Invalid page number in metadata"}, 500

        image_dir = os.path.join(self.image_directory, pdf_id)
        images = [f"{image_dir}/{pdf_id}_{i}.jpg" for i in range(page_nums)]
        return {"images": images}

    def serve(self):
        # keep this function to maintain compatibility with scripts/start_server.py
        print("Welcome to End-Of-Term PDF Search Server")

        print("Searching against " + str(self.faiss_index.ntotal) + " embeddings\n")
        try:
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
