from .config import ServerConfig
import numpy as np
import json
import faiss
import os
from .pdf_to_embedding import CLIPEmbeddingModel

class PrecomputedServer:
    def __init__(self, config: ServerConfig, embeddings_dir):
        self.config = config
        self.model = config.model
        self.k = self.config.k
        self.d = self.config.d
        self.embeddings_dir = embeddings_dir
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatL2(self.d)
        
        # Load embeddings from local disk
        print(f"Loading embeddings from {embeddings_dir}...")
        self.file_paths = []
        self.embeddings = []
        
        # Walk through the embeddings directory
        for root, _, files in os.walk(os.path.join(embeddings_dir, 'embeddings')):
            for file in files:
                if file.endswith('.npy'):
                    # Get relative path for the file ID
                    rel_path = os.path.relpath(os.path.join(root, file), 
                                              os.path.join(embeddings_dir, 'embeddings'))
                    # Remove .npy extension
                    file_id = os.path.splitext(rel_path)[0]
                    self.file_paths.append(file_id)
        
        # Load embeddings in batches
        batch_size = 1000
        for i in range(0, len(self.file_paths), batch_size):
            batch_paths = self.file_paths[i:i+batch_size]
            print(f"Loading batch {i//batch_size + 1}/{(len(self.file_paths)//batch_size) + 1}...")
            
            batch_embeddings = []
            for path in batch_paths:
                try:
                    embedding_path = os.path.join(embeddings_dir, 'embeddings', f"{path}.npy")
                    embedding = np.load(embedding_path)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error loading embedding for {path}: {e}")
            
            if batch_embeddings:
                stacked_array = np.vstack(batch_embeddings)
                self.faiss_index.add(stacked_array)
                self.embeddings.extend(batch_embeddings)
        
        print(f"Loaded {self.faiss_index.ntotal} embeddings into FAISS index")
    
    def serve(self):
        """Start the server and handle queries"""
        print("\nWelcome to End-Of-Term PDF Search Server (Precomputed Version)")
        print(f"Searching against {self.faiss_index.ntotal} embeddings\n")
        
        while True:
            try:
                print("Search: ", end="")
                query = input()
                if not query:
                    continue
                
                # Encode query
                query_embedding = self.model.encode_text(query)
                
                # Search for similar embeddings
                D, I = self.faiss_index.search(np.array([query_embedding]), self.k)
                
                # Prepare results
                search_results = []
                for i in range(len(I)):
                    for j in range(len(I[i])):
                        if I[i][j] < len(self.file_paths):
                            file_path = self.file_paths[I[i][j]]
                            
                            # Generate image URL (similar to S3 format)
                            jpeg_url = f"file://{os.path.join(self.config.image_directory, file_path)}_0.jpg"
                            
                            # Add to results
                            search_results.append({
                                "pdf": file_path,
                                "distance": float(D[i][j]),
                                "jpeg": jpeg_url
                            })
                
                # Output results as JSON
                json_object = json.dumps({"results": search_results}, indent=4)
                print(json_object)
                
            except KeyboardInterrupt:
                print("\nExiting server...")
                break
            except Exception as e:
                print(f"Error processing query: {e}") 