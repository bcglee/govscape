import os
import boto3
import numpy as np
import io
import time
from tqdm import tqdm

class S3DataLoader:
    def __init__(self, bucket_name="bcgl-public-bucket", base_path="2008_EOT_PDFs"):
        """Initialize S3 data loader with bucket and path information"""
        self.bucket_name = bucket_name
        self.base_path = base_path
        self.s3_client = boto3.client('s3')
        
    def list_processed_embeddings(self):
        """List all processed embeddings in S3"""
        try:
            # Get the manifest file that lists all PDFs
            manifest_key = f"{self.base_path}/manifest.txt"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=manifest_key)
            manifest_content = response['Body'].read().decode('utf-8')
            
            # Parse the manifest to get file paths
            file_paths = []
            for line in manifest_content.strip().split('\n'):
                file_path = line.strip()
                if file_path:
                    file_paths.append(file_path)
            
            print(f"Found {len(file_paths)} PDF files in manifest")
            return file_paths
        except Exception as e:
            print(f"Error reading manifest file: {e}")
            return []
    
    def load_embedding(self, file_path):
        """Load an embedding from S3"""
        try:
            embedding_key = f"{self.base_path}/metadata/{file_path}.npy"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=embedding_key)
            embedding_bytes = response['Body'].read()
            embedding = np.load(io.BytesIO(embedding_bytes))
            return embedding
        except Exception as e:
            print(f"Error loading embedding for {file_path}: {e}")
            return None
    
    def load_text(self, file_path):
        """Load text content from S3"""
        text_key = f"{self.base_path}/metadata/{file_path}.txt"
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=text_key)
            text = response['Body'].read().decode('utf-8')
            return text
        except Exception as e:
            print(f"Error loading text for {file_path}: {e}")
            return ""
    
    def get_image_url(self, file_path, page_num=0):
        """Generate URL for page image with correct format"""
        # Use 7-digit page number with underscore prefix
        page_str = f"_{page_num:07d}"
        image_key = f"{self.base_path}/metadata/{file_path}{page_str}.jpg"
        return f"https://{self.bucket_name}.s3.amazonaws.com/{image_key}"
    
    # New method for downloading embeddings to local disk
    def download_embedding(self, file_path, output_dir):
        """Download an embedding from S3 to local disk"""
        try:
            embedding = self.load_embedding(file_path)
            if embedding is not None:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.join(output_dir, file_path)), exist_ok=True)
                # Save embedding to local disk
                np.save(os.path.join(output_dir, f"{file_path}.npy"), embedding)
                return True
            return False
        except Exception as e:
            print(f"Error downloading embedding for {file_path}: {e}")
            return False
    
    # New method for downloading text to local disk
    def download_text(self, file_path, output_dir):
        """Download text content from S3 to local disk"""
        try:
            text = self.load_text(file_path)
            if text:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.join(output_dir, file_path)), exist_ok=True)
                # Save text to local disk
                with open(os.path.join(output_dir, f"{file_path}.txt"), 'w', encoding='utf-8') as f:
                    f.write(text)
                return True
            return False
        except Exception as e:
            print(f"Error downloading text for {file_path}: {e}")
            return False
    
    # New method for downloading images to local disk
    def download_image(self, file_path, page_num, output_dir):
        """Download page image from S3 to local disk"""
        try:
            # Use 7-digit page number with underscore prefix
            page_str = f"_{page_num:07d}"
            image_key = f"{self.base_path}/metadata/{file_path}{page_str}.jpg"
            
            # Create output path
            output_path = os.path.join(output_dir, f"{file_path}{page_str}.jpg")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download image
            self.s3_client.download_file(self.bucket_name, image_key, output_path)
            return True
        except Exception as e:
            print(f"Error downloading image for {file_path} page {page_num}: {e}")
            return False
    
    # New method for batch downloading embeddings
    def batch_download_embeddings(self, file_paths, output_dir, batch_size=100):
        """Download embeddings in batches with progress tracking"""
        os.makedirs(output_dir, exist_ok=True)
        
        total_files = len(file_paths)
        success_count = 0
        error_count = 0
        
        for i in range(0, total_files, batch_size):
            batch = file_paths[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_files//batch_size) + 1}...")
            
            start_time = time.time()
            progress_bar = tqdm(batch, desc="Downloading embeddings", unit="file")
            
            for file_path in progress_bar:
                success = self.download_embedding(file_path, output_dir)
                if success:
                    success_count += 1
                else:
                    error_count += 1
                
                # Update progress bar description
                progress_bar.set_description(
                    f"Success: {success_count}, Errors: {error_count}"
                )
            
            elapsed = time.time() - start_time
            rate = len(batch) / elapsed if elapsed > 0 else 0
            print(f"  Completed batch with {success_count - (i//batch_size * batch_size)} successful downloads and {error_count - (i//batch_size * batch_size)} errors")
            print(f"  Rate: {rate:.1f} files/sec")
        
        print(f"Download complete. Total: {total_files}, Success: {success_count}, Errors: {error_count}")
        return success_count, error_count
    
    # New method for batch downloading text files
    def batch_download_texts(self, file_paths, output_dir, batch_size=100):
        """Download text files in batches with progress tracking"""
        os.makedirs(output_dir, exist_ok=True)
        
        total_files = len(file_paths)
        success_count = 0
        error_count = 0
        
        for i in range(0, total_files, batch_size):
            batch = file_paths[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_files//batch_size) + 1}...")
            
            start_time = time.time()
            progress_bar = tqdm(batch, desc="Downloading text files", unit="file")
            
            for file_path in progress_bar:
                success = self.download_text(file_path, output_dir)
                if success:
                    success_count += 1
                else:
                    error_count += 1
                
                # Update progress bar description
                progress_bar.set_description(
                    f"Success: {success_count}, Errors: {error_count}"
                )
            
            elapsed = time.time() - start_time
            rate = len(batch) / elapsed if elapsed > 0 else 0
            print(f"  Completed batch with {success_count - (i//batch_size * batch_size)} successful downloads and {error_count - (i//batch_size * batch_size)} errors")
            print(f"  Rate: {rate:.1f} files/sec")
        
        print(f"Download complete. Total: {total_files}, Success: {success_count}, Errors: {error_count}")
        return success_count, error_count
    
    # New method for verifying downloaded files
    def verify_downloads(self, file_paths, output_dir):
        """Verify that downloaded files exist and are valid"""
        missing = []
        corrupted = []
        
        print(f"Verifying {len(file_paths)} downloaded files...")
        progress_bar = tqdm(file_paths, desc="Verifying files", unit="file")
        
        for file_path in progress_bar:
            embedding_path = os.path.join(output_dir, f"{file_path}.npy")
            
            # Check if file exists
            if not os.path.exists(embedding_path):
                missing.append(file_path)
                continue
            
            # Try to load the file to check if it's corrupted
            try:
                np.load(embedding_path)
            except Exception:
                corrupted.append(file_path)
        
        print(f"Verification complete. Missing: {len(missing)}, Corrupted: {len(corrupted)}")
        return missing, corrupted
    
    def verify_embeddings(self, n_samples=5):
        """Verify embeddings exist in S3"""
        file_paths = self.list_processed_embeddings()
        
        if not file_paths:
            print("No file paths found in manifest!")
            return
        
        print("\nVerifying sample embeddings...")
        for i in range(min(n_samples, len(file_paths))):
            file_path = file_paths[i]
            embedding_key = f"{self.base_path}/metadata/{file_path}.npy"
            
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=embedding_key)
                print(f"✅ {embedding_key} exists")
            except Exception as e:
                print(f"❌ {embedding_key} does not exist: {e}")
                
                try:
                    prefix = f"{self.base_path}/metadata/{file_path.split('_')[0]}"
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        Prefix=prefix,
                        MaxKeys=5
                    )
                    
                    if 'Contents' in response:
                        print("  Similar files found:")
                        for obj in response['Contents']:
                            print(f"  - {obj['Key']}")
                except Exception:
                    pass
    
    def load_embeddings_batch(self, file_paths, batch_start, batch_size):
        """Load a batch of embeddings"""
        batch_end = min(batch_start + batch_size, len(file_paths))
        batch_embeddings = []
        success_count = 0
        error_count = 0
        start_time = time.time()
        
        for i in range(batch_start, batch_end):
            file_path = file_paths[i]
            embedding = self.load_embedding(file_path)
            
            if embedding is not None:
                batch_embeddings.append(embedding)
                success_count += 1
            else:
                error_count += 1
                
            # Show detailed progress (every 20 files)
            if (i - batch_start + 1) % 20 == 0 or i == batch_end - 1:
                elapsed = time.time() - start_time
                rate = (i - batch_start + 1) / elapsed if elapsed > 0 else 0
                remaining = (batch_end - i - 1) / rate if rate > 0 else 0
                
                print(f"  Progress: {i - batch_start + 1}/{batch_end - batch_start} files | " 
                      f"Success: {success_count}, Errors: {error_count} | "
                      f"Rate: {rate:.1f} files/sec | "
                      f"Est. remaining: {int(remaining//60)}m {int(remaining%60)}s", end="\r")
        
        print(f"\n  Completed batch with {len(batch_embeddings)} successful embeddings and {error_count} errors")
        return batch_embeddings
    
    def verify_image_path(self, file_path, page_num=0):
        """Verify image path exists on S3"""
        page_str = f"_{page_num:07d}"
        image_key = f"{self.base_path}/metadata/{file_path}{page_str}.jpg"
        
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=image_key)
            print(f"✅ Image exists: {image_key}")
            return True
        except Exception as e:
            print(f"❌ Image does not exist: {image_key}")
            print(f"  Error: {e}")
            return False 