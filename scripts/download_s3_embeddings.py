import boto3
import os
import argparse

def download_from_s3(bucket_name, s3_prefix, local_dir):
    s3_client = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            s3_path = obj['Key']
            local_path = os.path.join(local_dir, s3_path.replace(s3_prefix, ''))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            s3_client.download_file(bucket_name, s3_path, local_path)
            print(f"Downloaded: {s3_path} -> {local_path}")

def main():
    parser = argparse.ArgumentParser(description='Download embeddings from S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', required=True, help='S3 prefix (folder path)')
    parser.add_argument('--local-dir', required=True, help='Local directory to save files')
    
    args = parser.parse_args()
    download_from_s3(args.bucket, args.prefix, args.local_dir)

if __name__ == '__main__':
    main()
