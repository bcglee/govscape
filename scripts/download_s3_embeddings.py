import boto3
import os
import argparse

def download_from_s3(bucket_name, prefix, local_dir):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    # Define folders to download
    folders = [
        'img',
        'txt',
        'img_extracted',
        'embeddings',
        'embeddings_img_pg',
        'embeddings_img_extracted'
    ]
    
    print("Downloading embeddings...")
    for folder in folders:
        s3_folder_path = f"{prefix}/{folder}/"
        print(f"\nDownloading {folder}...")
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder_path):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                s3_path = obj['Key']
                relative_path = s3_path[len(prefix)+1:]
                local_path = os.path.join(local_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                print(f"Downloading {s3_path} to {local_path}")
                s3_client.download_file(bucket_name, s3_path, local_path)

def main():
    parser = argparse.ArgumentParser(description='Download files from S3 bucket')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', required=True, help='S3 prefix (folder path)')
    parser.add_argument('--local-dir', required=True, help='Local directory to save files')
    
    args = parser.parse_args()
    download_from_s3(args.bucket, args.prefix, args.local_dir)

if __name__ == '__main__':
    main()
