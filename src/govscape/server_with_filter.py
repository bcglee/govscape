# This file defines the logic for serving requests to the user.
from .config import ServerConfig
from .config import IndexConfig
import numpy as np
import json
import faiss
import os
from .pdf_to_embedding import PDFsToEmbeddings


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

        # FAISS model
        self.model = config.model
        self.k = self.config.k
        self.d = self.config.d

        # create a new index
        self.faiss_index = faiss.IndexFlatL2(self.d)

        # Train model on test vectors
        self.npy_files = []
        self.page_mappings = {}
        for root, _, files in os.walk(self.embedding_directory):
            for file in files:
                if file.endswith('.npy'):
                    self.npy_files.append(os.path.join(root, file))
                if file.endswith('.json'):
                    with open(file, "r") as file_json:
                        data = json.load(file_json)
                        directory = os.path.dirname(file)
                        self.page_mappings[directory] = data["num_pages"]


        # Load each .npy file into an array
        self.arrays = [np.load(file) for file in self.npy_files]
        for array in self.arrays:
            print(array.shape)
        stacked_array = np.vstack(self.arrays) 
        self.faiss_index.add(stacked_array)
    
    def filter_results_num_pages(self, search_results, min_page, max_page):
        filtered_search_results = []
        i = 0
        for result in search_results:
            if i == self.k:
                break
            file = res["pdf"]
            directory = os.path.dirname(file)
            if min_page <= self.page_mappings[directory] <= max_page:
                filtered_search_results.append(res)
                i += 1
        return filtered_search_results


    # Accepts a Query -> Returns JSON with closest results
    # Sample:
    # {
    #     "results": [
    #         {
    #             "pdf": "test_data/embeddings/gold/gold",
    #             "page": "0",
    #             "distance": 59.212852478027344
    #         },
    #         {
    #             "pdf": "test_data/embeddings/government/government",
    #             "page": "0",
    #             "distance": 68.0333251953125
    #         },
    #         {s
    #             "pdf": "test_data/embeddings/joebiden/joebiden",
    #             "page": "0",
    #             "distance": 68.0333251953125
    #         }
    #     ]
    # }

    def serve(self):
        print("Welcome to End-Of-Term PDF Search Server")   
    
        print("Searching against " + str(self.faiss_index.ntotal) + " embeddings\n")
        try:
            while True:
                query = input("Search: ")
                # EOF detected
                if query == "":
                    continue
                min_page = int(input("Please enter minimum number of PDF pages: "))
                max_page = int(input("Please enter maximum number of PDF pages: "))

                # Create random array embedding
                query_embedding = self.model.text_to_embeddings(query)
                # Search for the k closest arrays
                D, I = self.faiss_index.search(query_embedding, self.k * 2)

                search_results = []
                for i in range(I.shape[0]):
                    for j in range(I.shape[1]):
                        # parse file information for page
                        pdf_name, _, page = self.npy_files[I[i][j]].rpartition('_')
                        page, _, _ = page.rpartition('.')
                        # create jpeg name
                        jpeg = self.image_directory + "/" + "/".join(pdf_name.rsplit("/", 2)[-2:]) + "_" + page + '.jpg'

                        # add results onto file
                        search_results.append({"pdf": pdf_name, "page": page, "distance": float(D[i][j]), "jpeg": jpeg})
                filtered_search_results = self.filter_results_num_pages(search_results, min_page, max_page)
                json_object = json.dumps({"results": filtered_search_results}, indent=4)

                # print for testing
                print(json_object)
        except EOFError:
            print("\nThank you for using!")
