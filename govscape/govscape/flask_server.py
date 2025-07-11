# This file defines the logic for serving requests to the user.
from .config import ServerConfig
from .config import IndexConfig
import numpy as np
import json
import faiss
import os
from .pdf_to_embedding import PDFsToEmbeddings
from urllib.parse import urlparse, parse_qs
from flask import Flask, request

class Server:

    # obtain all the setup information from configuration
    def __init__(self, config : ServerConfig):
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
        self.faiss_index = self.config.faiss_index

        # files generated by embeddings
        self.npy_files = self.config.npy_files

    # Accepts a Query -> Returns JSON with closest results
    # Sample:
    # (1, 512)
    # (1, 512)
    # (1, 512)
    # (1, 512)
    # Welcome to End-Of-Term PDF Search Server
    # Searching against 4 embeddings

    # Search: /static?pdf_name=gold.pdf
    # {
    #     "pdf": [
    #         "gold.pdf"
    #     ]
    # }
    # Search: /search?query=joe
    # {
    #     "results": [
    #         {
    #             "pdf": "test_data/embeddings/gold/gold",
    #             "page": "0",
    #             "distance": 61.45836639404297,
    #             "jpeg": "test_data/imagesgold/gold_0"
    #         },
    #         {
    #             "pdf": "test_data/embeddings/government/government",
    #             "page": "0",
    #             "distance": 62.02237319946289,
    #             "jpeg": "test_data/imagesgovernment/government_0"
    #         },
    #         {
    #             "pdf": "test_data/embeddings/joebiden/joebiden",
    #             "page": "0",
    #             "distance": 62.02237319946289,
    #             "jpeg": "test_data/imagesjoebiden/joebiden_0"
    #         }
    #     ]
    # }

    def serve(self, query):
        if request == "":
            return json.dumps({"pdf": ""}, indent=4)
        
        query_embedding = self.model.text_to_embeddings(query)

        D, I = self.faiss_index.search(query_embedding, self.k)

        search_results = []
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                # parse file information for page
                pdf_name, _, page = self.npy_files[I[i][j]].rpartition('_')
                page, _, _ = page.rpartition('.')
                # create jpeg name
                jpeg = self.image_directory + "/".join(pdf_name.rsplit("/", 2)[-2:]) + "_" + page

                # add results onto file
                search_results.append({"pdf": pdf_name, "page": page, "distance": float(D[i][j]), "jpeg": jpeg})
        json_object = json.dumps({"results": search_results}, indent=4)
        return json_object

s = None
index_config = None
server_config = None

# disabling static functionality so we can have static endpoint
app = Flask(__name__, static_folder=None)

def initialize_server():
    global s
    global index_config
    global server_config
    index_config = IndexConfig()
    server_config = ServerConfig(index_config, PDFsToEmbeddings('test_data/pdfs', 'test_data/text', 'test_data/embeddings'))
    s = Server(server_config)

# set up server on opening
with app.app_context():
    initialize_server()

# home base
@app.route("/")
def home():
    return "Welcome to Govscape Search"

# search functionality
@app.route("/search")
def search():
    query = request.args.get('query')
    if query == "":
        return query
    else:
        return s.serve(query)

# static functionality
@app.route("/static")
def static():
    filename = request.args.get('pdf')
    fullfilename = os.path.join(index_config.pdf_directory, filename)
    if filename in os.listdir(index_config.pdf_directory):
        json_object = json.dumps({"pdf": fullfilename}, indent=4)
        return json_object
    else:
        json_object = json.dumps({}, indent=4)
        return json_object