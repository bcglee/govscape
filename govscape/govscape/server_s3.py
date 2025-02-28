from .config import ServerConfig
import numpy as np
import json
import faiss
import os
from .pdf_to_embedding import CLIPEmbeddingModel
from .s3_loader import S3DataLoader

class S3Server:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = config.model
        self.k = self.config.k
        self.d = self.config.d
        
        self.s3_loader = S3DataLoader()
        
        self.faiss_index = faiss.IndexFlatL2(self.d)
        
        print("Loading embeddings from S3...")
        self.file_paths = self.s3_loader.list_processed_embeddings()
        
        batch_size = 1000
        self.embeddings = []
        
        for i in range(0, len(self.file_paths), batch_size):
            batch_paths = self.file_paths[i:i+batch_size]
            print(f"Loading batch {i//batch_size + 1}/{(len(self.file_paths)//batch_size) + 1}...")
            
            batch_embeddings = []
            for path in batch_paths:
                embedding = self.s3_loader.load_embedding(path)
                if embedding is not None:
                    batch_embeddings.append(embedding)
            
            if batch_embeddings:
                stacked_array = np.vstack(batch_embeddings)
                self.faiss_index.add(stacked_array)
                self.embeddings.extend(batch_embeddings)
        
        print(f"Loaded {self.faiss_index.ntotal} embeddings into FAISS index")
    
    def serve(self):
        print("Welcome to End-Of-Term PDF Search Server (S3 Version)")   
        print(f"Searching against {self.faiss_index.ntotal} embeddings\n")
        
        try:
            while True:
                query = input("Search: ")
                if query == "":
                    continue
                
                query_embedding = self.model.encode_text(query)
                
                D, I = self.faiss_index.search(np.array([query_embedding]), self.k)
                
                search_results = []
                for i in range(I.shape[0]):
                    for j in range(I.shape[1]):
                        if I[i][j] < len(self.file_paths):
                            file_path = self.file_paths[I[i][j]]
                            
                            jpeg_url = self.s3_loader.get_image_url(file_path)
                            
                            search_results.append({
                                "pdf": file_path,
                                "distance": float(D[i][j]),
                                "jpeg": jpeg_url
                            })
                
                json_object = json.dumps({"results": search_results}, indent=4)
                print(json_object)
        
        except EOFError:
            print("\nThank you for using!") 