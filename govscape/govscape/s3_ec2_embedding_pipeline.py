# ***************************************************************************************************************************************
# this file is for when we are running the embedding pipeline on the pdfs in an aws ec2 instance. 
# we will be taking pdfs from s3 bucket and putting back the generated data (embeddings, images, etc.) back onto the s3 bucket. 
# process:
# 1. get batch of PDFS from s3 of batch_size on to the ec2 instance. 
# 2. process batch and generate embeddings, etc. 
# 3. put resulting data folder back to the s3 bucket. 
# 4. repeat until we've process all pdfs 
# ***************************************************************************************************************************************

# pip installs: 
# pip install boto3

import boto3
import os
import argparse
import govscape as gs

s3 = boto3.client('s3')

# FIELDS TO SET **************************************************************************************

BATCH_SIZE = 1000

# s3://bcgl-public-bucket/2008_EOT_PDFs/PDFs/
bucket_name = 'bcgl-public-bucket'
pdfs_dir = '2008_EOT_PDFs/PDFs/'
data_dir_s3 = '2008_EOT_PDFs/data1/'


# for processing pdfs: 
# pdf_directory = '../../data/test_data/small_test'
# txt_directory = '../../data/test_data/txt'
# embeddings_directory = '../../data/test_data/embeddings'
# image_directory = '../../data/test_data/images'
# model = gs.TextEmbeddingModel()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'test_data')

pdf_directory = os.path.join(DATA_DIR, 'small_test')
txt_directory = os.path.join(DATA_DIR, 'txt')
embeddings_directory = os.path.join(DATA_DIR, 'embeddings')
image_directory = os.path.join(DATA_DIR, 'images')

# processor = gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, image_directory, model)

# ****************************************************************************************************

def process_pdfs(pdf_files):
    #for pdf in pdf_files:
        # PROCESS THEM HERE
        # processor.pdfs_to_embeddings()
    # pdftojpeg = gs.PdfToJpeg(pdf_directory, image_directory, 100)
    # pdftojpeg.convert_directory_to_jpegs()

    upload_directory_to_s3(txt_directory, data_dir_s3 + 'txt')
    upload_directory_to_s3(embeddings_directory, data_dir_s3 + 'embeddings')
    upload_directory_to_s3(image_directory, data_dir_s3 + 'images')

def batched_file_download(BATCH_SIZE):
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
    # get list of pdf file names
    pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]

    # now process file batch by batch
    for i in range(0, len(pdf_files), BATCH_SIZE):
        batch = pdf_files[i:i + BATCH_SIZE]

        # Download each file in the batch
        for pdf in batch:
            file_name = os.path.basename(pdf)

            local_path = os.path.join('downloads', file_name)  # Save to 'downloads' folder
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            #download file into instance folder 'downloads'
            s3.download_file(bucket_name, pdf, local_path)
        
        process_pdfs(batch)
        break # TODO: remove this later, just testing the first batch if that works *************************************************

def upload_directory_to_s3(ec2_dir, s3_dir):
    for root, dirs, files in os.walk(ec2_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.join(s3_dir, os.path.relpath(local_file_path, ec2_dir)).replace("\\", "/")

            print(f"uploading {local_file_path} to {s3_key}...")
            s3.upload_file(local_file_path, bucket_name, s3_key)


#poetry run python s3_ec2_embedding_pipeline.py
def main():
    batched_file_download(BATCH_SIZE)

if __name__ == '__main__':
    main()