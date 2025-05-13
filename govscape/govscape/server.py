# This file defines the logic for serving requests to the user.
from .config import ServerConfig
from .config import IndexConfig
import numpy as np
import json
import faiss
import os
from .pdf_to_embedding import PDFsToEmbeddings
from urllib.parse import urlparse, parse_qs
# from whoosh.fields import Schema, TEXT, ID
# from whoosh.index import create_in, open_dir
# from whoosh.qparser import QueryParser, MultifieldParser, OrGroup, AndGroup


# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize FAISS to do this
class Server:

    # obtain all the setup information from configuration
    def __init__(self, config : ServerConfig):
        self.config = config

        # directories
        self.pdf_directory = config.pdf_directory
        self.embedding_directory = config.embedding_directory
        self.index_directory = config.index_directory
        self.image_directory = config.image_directory
        self.whoosh_directory = config.whoosh_directory

        # FAISS model
        self.model = config.model
        self.k = self.config.k
        self.d = self.config.d

        # create a new index
        self.faiss_index = faiss.IndexFlatL2(self.d)
        self.faiss_image_index = faiss.IndexFlatL2(self.d)

        # Train model on test vectors
        self.text_files = []
        self.img_files = []
        for root, _, files in os.walk(self.embedding_directory):
            for file in files:
                if file.endswith('.npy'):
                    full_path = os.path.join(root, file)
                    if 'text' in root:
                        self.text_files.append(full_path)
                    elif 'img' in root:
                        self.img_files.append(full_path)

        # Load each .npy file into an array
        self.text_arrays = [np.load(file) for file in self.text_files]
        self.img_arrays = [np.load(file) for file in self.img_files]


        text_stacked = np.vstack(self.text_arrays)
        img_stacked = np.vstack(self.img_arrays)
        self.faiss_index.add(text_stacked)
        self.faiss_image_index.add(img_stacked)
        # self.ix = open_dir(self.whoosh_directory)


        
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
    #         {s
    #             "pdf": "test_data/embeddings/joebiden/joebiden",
    #             "page": "0",
    #             "distance": 62.02237319946289,
    #             "jpeg": "test_data/imagesjoebiden/joebiden_0"
    #         }
    #     ]
    # }

    def serve(self):
        print("Welcome to End-Of-Term PDF Search Server")   
    
        print("Searching against " + str(self.faiss_index.ntotal) + " embeddings\n")
        try:
            while True:
                request = input("Search: ")
                # EOF detected
                if request == "":
                    continue

                parsed_url = urlparse(request)
                # Extract path and query parameters
                path = parsed_url.path
                query_params = parse_qs(parsed_url.query)

                json_object = json.dumps({})
                if 'static' in path:
                    # stores pdf name locally
                    pdf_name = query_params['pdf_name'][0]
                    npy_files = []
                    # searches the embedding directory
                    for entry in os.listdir(self.embedding_directory):
                        full_path = os.path.join(self.embedding_directory, entry)
                        if os.path.isdir(full_path) and pdf_name.startswith(entry):
                            for f in os.listdir(full_path):
                                if f.endswith('.npy'):
                                    npy_files.append(os.path.splitext(f)[0] + ".jpg")
                    json_object = json.dumps({"pdf": query_params['pdf_name'], "pages": npy_files}, indent=4)

                elif 'search' in path:
                    if 'exact' in query_params and (query_params['exact'][0].lower() == "true"):
                        with self.ix.searcher() as searcher:
                            parser = QueryParser("content", self.ix.schema)
                            query = parser.parse(query_params['query'][0])
                            results = searcher.search(query, limit = self.k)
                            search_results = []
                            for result in results:
                                search_results.append({"pdf": result['path']})
                            json_object = json.dumps({"results": search_results}, indent=4)
                    elif 'image' in query_params and (query_params['image'][0].lower() == "true"):
                        # Create random array embedding
                        query_embedding = self.model.text_to_embeddings(query_params['query'])

                        # Search for the three closest arrays
                        D, I = self.faiss_image_index.search(query_embedding, self.k)

                        search_results = []
                        for i in range(I.shape[0]):
                            for j in range(I.shape[1]):
                                # parse file information for page
                                pdf_name, _, page = self.img_files[I[i][j]].rpartition('_')
                                page, _, _ = page.rpartition('.')
                                # create jpeg name
                                jpeg = self.image_directory + "/".join(pdf_name.rsplit("/", 2)[-2:]) + "_" + page

                                # add results onto file
                                search_results.append({"pdf": pdf_name, "page": page, "distance": float(D[i][j]), "jpeg": jpeg})
                        json_object = json.dumps({"results": search_results}, indent=4)
                    else:
                        # Create random array embedding
                        query_embedding = self.model.text_to_embeddings(query_params['query'])

                        # Search for the three closest arrays
                        D, I = self.faiss_index.search(query_embedding, self.k)

                        search_results = []
                        for i in range(I.shape[0]):
                            for j in range(I.shape[1]):
                                # parse file information for page
                                pdf_name, _, page = self.text_files[I[i][j]].rpartition('_')
                                page, _, _ = page.rpartition('.')
                                # create jpeg name
                                jpeg = self.image_directory + "/".join(pdf_name.rsplit("/", 2)[-2:]) + "_" + page

                                # add results onto file
                                search_results.append({"pdf": pdf_name, "page": page, "distance": float(D[i][j]), "jpeg": jpeg})
                        json_object = json.dumps({"results": search_results}, indent=4)

                # print for testing
                print(json_object)
        except EOFError:
            print("\nThank you for using!")

if __name__ == '__main__':
         main()

'''
Test: EOF
end of file input (^D)

Test: empty input in url
/

Test: empty input in params
/static?pdf_name=
/search?

Test: regular input in search
/search?query=joe

Test: regular input in static
/static?pdf_name=gold.pdf

Test: non existent pdf in static
/static?pdf_name=fake.pdf

'''
