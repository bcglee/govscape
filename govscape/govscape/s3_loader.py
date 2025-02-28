import os
import boto3
import numpy as np
import io

class S3DataLoader:
    def __init__(self, bucket_name="bcgl-public-bucket", base_path="2008_EOT_PDFs"):
        self.bucket_name = bucket_name
        self.base_path = base_path
        self.s3_client = boto3.client('s3')
        
    def list_processed_embeddings(self, manifest_path="logs/100k_processed_to_image.csv"):
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"{self.base_path}/{manifest_path}"
            )
            
            content = response['Body'].read().decode('utf-8')
            file_paths = []
            
            lines = content.split('\n')
            header_processed = False
            
            for line in lines:
                if not line.strip():
                    continue
                    
                if not header_processed:
                    header_processed = True
                    continue
                
                parts = line.split(',', 1)
                if not parts:
                    continue
                    
                file_path = parts[0].strip()
                
                if file_path.lower().endswith('.pdf'):
                    file_path = file_path[:-4]
                    
                file_paths.append(file_path)
            
            print(f"Found {len(file_paths)} PDF files in manifest")
            return file_paths
        except Exception as e:
            print(f"Error reading manifest file: {e}")
            return []
    
    def load_embedding(self, file_path):
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
        text_key = f"{self.base_path}/metadata/{file_path}.txt"
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=text_key)
            text = response['Body'].read().decode('utf-8')
            return text
        except Exception as e:
            print(f"Error loading text for {file_path}: {e}")
            return ""
    
    def get_image_url(self, file_path, page_num=0):
        page_str = f"{page_num:08d}"
        image_key = f"{self.base_path}/metadata/{file_path}{page_str}.jpg"
        return f"https://{self.bucket_name}.s3.amazonaws.com/{image_key}"
    
    def verify_embeddings(self, n_samples=5):
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