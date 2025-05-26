import boto3
import os
import argparse
import time
import subprocess
import govscape as gs
import concurrent.futures
import shutil
import json

s3 = boto3.client('s3')

# FIELDS TO SET **************************************************************************************

BATCH_SIZE = 10 #TODO: FIX TO 1000

# s3://bcgl-public-bucket/2008_EOT_PDFs/PDFs/
bucket_name = 'bcgl-public-bucket'
pdfs_dir = '2008_EOT_PDFs/PDFs/'
data_dir_s3 = '2008_EOT_PDFs/data_test_100k/' # OUTPUT OVERALL DATA DIR IN S3 HERE  # TODO: CHANGE THE NAME OF DATA5
# data_dir_s3 = '2008_EOT_PDFs/data_test_100k_single_gpu_1/' # OUTPUT OVERALL DATA DIR IN S3 HERE  # TODO: CHANGE THE NAME OF DATA5
# data and data1 were for testing cpu file output
# data2 is for testing single gpu file output

# for processing pdfs: 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'test_data')  # THIS IS WHERE THE OVERALL DATA DIR IS IN EC2

#pdf_directory = os.path.join(DATA_DIR, 'small_test')  # for testing a folder of three pdfs
#pdf_directory = os.path.join(DATA_DIR, 'TechnicalReport234PDFs')
pdf_directory = 'downloads'
txt_directory = os.path.join(DATA_DIR, 'txt')
image_directory = os.path.join(DATA_DIR, 'img')
img_extracted_dir = os.path.join(DATA_DIR, 'img_extracted')
embeddings_directory = os.path.join(DATA_DIR, 'embeddings')
img_embeddings_dir = os.path.join(DATA_DIR, 'embeddings_img_pg')
e_img_embed_dir = os.path.join(DATA_DIR, 'embeddings_img_extracted')

batch_download_dir = 'downloads'  # temporary downloading (not sure if actually temp)

# model = gs.TextEmbeddingModel()
processor = gs.PDFsToEmbeddings(pdf_directory, txt_directory, image_directory, img_extracted_dir, 
                                embeddings_directory, img_embeddings_dir, e_img_embed_dir) # removed model


progress_path = 'progress.json'

# ****************************************************************************************************

# for analyzing: 
pipeline_times = {'first': 0, 'second': 0, 'third': 0, 'fourth': 0, 'fifth' : 0}

def get_n_pdfs(limit=100000):
    pdf_files = []
    continuation_token = None

    while True:
        if continuation_token:
            result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir, ContinuationToken=continuation_token)
        else:
            result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)

        contents = result.get('Contents', [])
        pdf_keys = [obj['Key'] for obj in contents if obj['Key'].endswith('.pdf')]

        remaining = limit - len(pdf_files)
        pdf_files.extend(pdf_keys[:remaining])

        if len(pdf_files) >= limit:
            if result.get('IsTruncated'):
                next_token = result.get('NextContinuationToken')
                with open(progress_path, 'w') as f:
                    json.dump({'continuation_token': next_token}, f)
            break

        if result.get('IsTruncated'):
            continuation_token = result.get('NextContinuationToken')
        else:
            break

    return pdf_files

def upload_directory_to_s3(ec2_dir, s3_dir):
    for root, dirs, files in os.walk(ec2_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.join(s3_dir, os.path.relpath(local_file_path, ec2_dir)).replace("\\", "/")
            s3.upload_file(local_file_path, bucket_name, s3_key)

def process_pdfs(pdf_files, processor):
    print("IN PROCESS_PDFS: ", pdf_files)
    start_time = time.time()

    # PROCESS PDFS HERE 
    one, two, three, four, five = processor.pdfs_to_embeddings(pdf_files=pdf_files)
    pipeline_times['first'] += one
    pipeline_times['second'] += two
    pipeline_times['third'] += three 
    pipeline_times['fourth'] += four
    pipeline_times['fifth'] += five

    end_time = time.time()
    duration = end_time - start_time
    if duration > 0:
        throughput = len(pdf_files) / duration
    else:
        throughput = 0

    # UPLOADING EMBEDDINGS, TXTS, IMAGES TO S3 HERE 
    upload_directory_to_s3(txt_directory, data_dir_s3 + 'txt')
    upload_directory_to_s3(image_directory, data_dir_s3 + 'img')
    upload_directory_to_s3(img_extracted_dir, data_dir_s3 + 'img_extracted')
    upload_directory_to_s3(embeddings_directory, data_dir_s3 + 'embeddings')
    upload_directory_to_s3(img_embeddings_dir, data_dir_s3 + 'embeddings_img_pg')
    upload_directory_to_s3(e_img_embed_dir, data_dir_s3 + 'embeddings_img_extracted')

    print("finished uploading current batch")


# NON-MULTITHREADED VERSION
def batched_file_download(BATCH_SIZE, processor):
    # result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
    # # Get list of pdf file names
    # pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000

    # temporary: 
    # pdf_files = pdf_files[0:10]

    pdf_files = get_n_pdfs()

    print("Now starting with total number of PDF files: ", len(pdf_files))
    
    overall_start_time = time.time()

    a = 0
    for i in range(0, len(pdf_files), BATCH_SIZE):
        print("BATCH: ", i)
        batch = pdf_files[i:i + BATCH_SIZE] 
        local_batch = []

        for pdf in batch:
            file_name = os.path.basename(pdf)
            local_path = os.path.join('downloads', file_name)  # save here?
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket_name, pdf, local_path)
            local_batch.append(file_name)
        
        process_pdfs(local_batch, processor)  #TODO: ?? 

        # TODO: DELTE THE TXT FOLDERS AND OTHERS 
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        if os.path.exists(pdf_directory):
            shutil.rmtree(pdf_directory)
        
        a = a + 1
        if a == 1:
            break
    
    overall_end_time = time.time()

    print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))
    print("TOTAL TIME pdf -> txt time:",  pipeline_times['first'])
    print("TOTAL TIME txt -> embed time:", pipeline_times['second'])
    print("TOTAL TIME pdf -> img per page time:", pipeline_times['third'])
    print("TOTAL TIME img per page -> embed time:", pipeline_times['fourth'])
    print("TOTAL TIME img extracted -> embed time :", pipeline_times['fifth'])

#poetry run python s3_ec2_embedding_pipeline.py
def main():
    # model = gs.TextEmbeddingModel()
    # processor = gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, image_directory, model)
    batched_file_download(BATCH_SIZE, processor)

if __name__ == '__main__':
    main()
