import boto3
import os
import argparse
import time
import subprocess
import govscape as gs
import concurrent.futures

s3 = boto3.client('s3')

# FIELDS TO SET **************************************************************************************

BATCH_SIZE = 50

# s3://bcgl-public-bucket/2008_EOT_PDFs/PDFs/
bucket_name = 'bcgl-public-bucket'
pdfs_dir = '2008_EOT_PDFs/PDFs/'
data_dir_s3 = '2008_EOT_PDFs/data3/' # OUTPUT OVERALL DATA DIR IN S3 HERE
# data and data1 were for testing cpu file output
# data2 is for testing single gpu file output

# for processing pdfs: 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'test_data')  # THIS IS WHERE THE OVERALL DATA DIR IS IN EC2

#pdf_directory = os.path.join(DATA_DIR, 'small_test')  # for testing a folder of three pdfs
#pdf_directory = os.path.join(DATA_DIR, 'TechnicalReport234PDFs')
pdf_directory = 'downloads'
txt_directory = os.path.join(DATA_DIR, 'txt')
embeddings_directory = os.path.join(DATA_DIR, 'embeddings')
image_directory = os.path.join(DATA_DIR, 'images')

batch_download_dir = 'downloads'  # temporary downloading (not sure if actually temp)

model = gs.TextEmbeddingModel()

processor = gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, image_directory, model)

# ****************************************************************************************************

# for analyzing: 
times = []

def upload_directory_to_s3(ec2_dir, s3_dir):
    for root, dirs, files in os.walk(ec2_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.join(s3_dir, os.path.relpath(local_file_path, ec2_dir)).replace("\\", "/")
            s3.upload_file(local_file_path, bucket_name, s3_key)

def process_pdfs(pdf_files):
    print("IN PROCESS_PDFS: ", pdf_files)
    start_time = time.time()

    # PROCESS PDFS HERE 
    processor.pdfs_to_embeddings(pdf_files=pdf_files)

    end_time = time.time()
    duration = end_time - start_time
    if duration > 0:
        throughput = len(pdf_files) / duration
    else:
        throughput = 0

    # UPLOADING EMBEDDINGS, TXTS, IMAGES TO S3 HERE 
    # upload_directory_to_s3(txt_directory, data_dir_s3 + 'txt')
    # upload_directory_to_s3(embeddings_directory, data_dir_s3 + 'embeddings')
    # upload_directory_to_s3(image_directory, data_dir_s3 + 'images')  # not working yet or idk
    print("finished uploading current batch")


# NON-MULTITHREADED VERSION
# def batched_file_download(BATCH_SIZE):
#     result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
#     # Get list of pdf file names
#     pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000

#     # temporary: 
#     pdf_files = pdf_files[0:100]

#     print("Now starting with total number of PDF files: ", len(pdf_files))
    

#     for i in range(0, len(pdf_files), BATCH_SIZE):
#         print("BATCH: ", i)
#         batch = pdf_files[i:i + BATCH_SIZE] 

#         for pdf in batch:
#             file_name = os.path.basename(pdf)
#             local_path = os.path.join('downloads', file_name)  # save here?
#             os.makedirs(os.path.dirname(local_path), exist_ok=True)
#             s3.download_file(bucket_name, pdf, local_path)
        
#         process_pdfs(batch)

# MULTITHREADED VERSION??? -- idk if this is really speeding up anything
def download_pdf(pdf, batch_download_dir):
    file_name = os.path.basename(pdf)
    local_path = os.path.join(batch_download_dir, file_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket_name, pdf, local_path)
    return local_path

def batched_file_download(BATCH_SIZE):
    # downloading pdfs from s3
    start_time_load_files = time.time()
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
    pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # YOU ARE ONLY GETTING THE FIRST 1000, have to do something with pages??

    pdf_files = pdf_files[:1000] # temporary!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! this is just for testing 

    print("TOTAL NUMBER OF PDF FILES WE ARE ABOUT TO PROCESS IS ", len(pdf_files))
    print("NUMBER OF THREADS FOR DOWNLOADING FILES IS ", os.cpu_count())

    local_pdf_files = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor: # TODO: maybe take thsi out 
        futures = []
        for pdf in pdf_files:
            futures.append(executor.submit(download_pdf, pdf, batch_download_dir))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                local_path = future.result()
                local_pdf_files.append(os.path.basename(local_path))
            except Exception as e:
                print(f"Download failed with error: {e}")
    
    print("LOCAL PDF FILES: ", local_pdf_files)

    end_time_load_files = time.time()

    start_time_process_files = time.time()
    # processing pdfs here in batches 
    for i in range(0, len(local_pdf_files), BATCH_SIZE):
        batch = local_pdf_files[i:i + BATCH_SIZE]
        process_pdfs(batch)
    end_time_process_files = time.time()
    
    time_load = end_time_load_files - start_time_load_files
    time_process = end_time_process_files - start_time_process_files

    print("TOTAL TIME TO LOAD IS ", time_load)
    print("TOTAL TIME TO PROCESS IS ", time_process)
    print("TOTAL TIME IS ", (time_process + time_load))

#poetry run python s3_ec2_embedding_pipeline.py
def main():
    batched_file_download(BATCH_SIZE)

if __name__ == '__main__':
    main()
