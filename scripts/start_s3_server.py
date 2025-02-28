import govscape as gs
import argparse
import numpy as np
import torch
from govscape.pdf_to_embedding import CLIPEmbeddingModel

def load_clip():
    print("Loading CLIP model...")
    model = CLIPEmbeddingModel()
    return model, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--verify', action='store_true', help='Verify S3 embedding files before starting server')
    parser.add_argument('--verify-image', action='store_true', help='Verify S3 image paths')
    parser.add_argument('--quick-test', '-q', action='store_true', help='Run in quick test mode with limited data')
    parser.add_argument('--limit', '-l', type=int, default=1000, help='Limit number of embeddings to load (for testing)')
    args = parser.parse_args()

    if args.verify:
        s3_loader = gs.S3DataLoader()
        s3_loader.verify_embeddings(n_samples=10)
        return
    
    if args.verify_image:
        s3_loader = gs.S3DataLoader()
        file_paths = s3_loader.list_processed_embeddings()
        if file_paths:
            # Verify first 5 files
            for i in range(min(5, len(file_paths))):
                s3_loader.verify_image_path(file_paths[i])
        return

    # Load CLIP model
    clip_model, preprocess = load_clip()
    
    # Load file list and embeddings
    s3_loader = gs.S3DataLoader()
    file_paths = s3_loader.list_processed_embeddings()
    
    # Limit files for quick testing
    limit = args.limit if args.quick_test else None
    if limit:
        print(f"QUICK TEST MODE: Only loading {limit} embeddings")
        file_paths = file_paths[:limit]
    
    # Create FAISS index and add embeddings
    index_config = gs.IndexConfig()
    server_config = gs.ServerConfig(index_config, clip_model, k=args.top_k)
    server = gs.S3Server(server_config)
    
    server.serve()

if __name__ == "__main__":
    main() 