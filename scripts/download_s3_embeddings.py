import govscape as gs
import argparse
import os
import numpy as np
import tqdm

def main():
    parser = argparse.ArgumentParser(
        prog='S3EmbeddingDownloader',
        description='Downloads embeddings from S3 to local disk',
        epilog='')
    parser.add_argument('-o', '--output-directory', required=True, 
                        help='Directory to save downloaded embeddings')
    parser.add_argument('-b', '--batch-size', type=int, default=1000, 
                        help='Batch size for downloading')
    parser.add_argument('--metadata', action='store_true',
                        help='Also download metadata files (text content)')
    args = parser.parse_args()
    
    output_dir = args.output_directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'embeddings'), exist_ok=True)
    if args.metadata:
        os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)
    
    # Initialize S3 loader
    s3_loader = gs.S3DataLoader()
    
    # Get list of embeddings
    print("Listing embeddings from S3...")
    file_paths = s3_loader.list_processed_embeddings()
    print(f"Found {len(file_paths)} embeddings")
    
    # Download embeddings in batches
    batch_size = args.batch_size
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i+batch_size]
        print(f"Downloading batch {i//batch_size + 1}/{(len(file_paths)//batch_size) + 1}...")
        
        for path in tqdm.tqdm(batch_paths):
            # Download embedding
            embedding = s3_loader.load_embedding(path)
            if embedding is not None:
                # Save embedding to local disk
                output_path = os.path.join(output_dir, 'embeddings', f"{path}.npy")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, embedding)
                
                # Download text content if requested
                if args.metadata:
                    text = s3_loader.load_text(path)
                    if text:
                        text_path = os.path.join(output_dir, 'text', f"{path}.txt")
                        os.makedirs(os.path.dirname(text_path), exist_ok=True)
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(text)
    
    print(f"Downloaded embeddings to {output_dir}")

if __name__ == '__main__':
    main() 