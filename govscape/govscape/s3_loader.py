import os
import boto3
import numpy as np
import io
import time

class S3DataLoader:
    def __init__(self, bucket_name="bcgl-public-bucket", base_path="2008_EOT_PDFs"):
        self.bucket_name = bucket_name
        self.base_path = base_path
        self.s3_client = boto3.client('s3')
        
    def list_processed_embeddings(self, manifest_path="logs/100k_processed_to_image.csv"):
        """Get list of processed embeddings"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"{self.base_path}/{manifest_path}"
            )
            
            content = response['Body'].read().decode('utf-8')
            file_paths = []
            
            # Process each line
            lines = content.split('\n')
            header_processed = False
            
            for line in lines:
                if not line.strip():
                    continue
                    
                # Skip header line (PDF_path,n_pages)
                if not header_processed:
                    header_processed = True
                    continue
                
                # Split each line, taking only the first part (PDF path)
                parts = line.split(',', 1)
                if not parts:
                    continue
                    
                file_path = parts[0].strip()
                
                # Remove .pdf extension
                if file_path.lower().endswith('.pdf'):
                    file_path = file_path[:-4]
                    
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