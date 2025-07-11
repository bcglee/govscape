import boto3
import os
import argparse
import time
import govscape as gs
import torch
import shutil
import json
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed

# ****************************************************************************************************
# to run this file: poetry run python s3_ec2_embedding_pipeline.py
# ****************************************************************************************************

if __name__ == '__main__':
    config = Config(max_pool_connections=50)
    s3 = boto3.client("s3", config=config)

    # FIELDS TO SET **************************************************************************************

    BATCH_SIZE = 1000 # number of files we are processing at a time

    # s3://bcgl-public-bucket/2008_EOT_PDFs/PDFs/
    bucket_name = 'bcgl-public-bucket'
    pdfs_dir = '2008_EOT_PDFs/PDFs/'
    data_dir_s3 = '2008_EOT_PDFs/data_test_100k_final/' # OUTPUT OVERALL DATA DIR IN S3 HERE 
    # data and data1 were for testing cpu file output
    # data2 is for testing single gpu file output
    # data_test_100k_final is for the final 50k

    # for processing pdfs: 

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'test_data')  # THIS IS WHERE THE OVERALL DATA DIR IS IN EC2

    pdf_directory = os.path.join(DATA_DIR, 'PDFs')
    txt_directory = os.path.join(DATA_DIR, 'txt')
    image_directory = os.path.join(DATA_DIR, 'img')
    img_extracted_dir = os.path.join(DATA_DIR, 'img_extracted')
    embeddings_directory = os.path.join(DATA_DIR, 'embeddings')
    img_embeddings_dir = os.path.join(DATA_DIR, 'embeddings_img_pg')
    e_img_embed_dir = os.path.join(DATA_DIR, 'embeddings_img_extracted')
    metadata_dir = os.path.join(DATA_DIR, 'metadata')

    text_model = gs.TextEmbeddingModel()
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append("cuda:" + str(i))
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    model_pool = text_model.model.start_multi_process_pool(target_devices=devices)
    processor = gs.PDFsToEmbeddings(pdf_directory, txt_directory, image_directory, img_extracted_dir, 
                                    embeddings_directory, img_embeddings_dir, e_img_embed_dir, metadata_dir, text_model, model_pool)

    progress_path = 'progress.json'  # when downloading files, keeps track of which page you last downloaded so you can resume later. haven't used this yet

    # ****************************************************************************************************

    # for analyzing: 
    pipeline_times = {'list' : 0, 'download' : 0, 'first': 0, 'second': 0, 'third': 0, 'fourth': 0, 'fifth' : 0, 'sixth': 0}  # to keep track of the time it takes for each step in the pipeline

    # gets pdfs from s3
    def get_n_pdfs(num_pages=1):
        pdf_files = []
        continuation_token = None
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                continuation_token = progress.get('continuation_token', None)
        pages_retrieved = 0
        while True:
            if continuation_token:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir, ContinuationToken=continuation_token)
            else:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
            
            contents = result.get('Contents', [])
            pdf_keys = [obj['Key'] for obj in contents if obj['Key'].endswith('.pdf')]

            pdf_files.extend(pdf_keys)
            pages_retrieved += 1
            if result.get('IsTruncated'):
                continuation_token = result.get('NextContinuationToken')
                with open(progress_path, 'w') as f:
                    json.dump({'continuation_token': continuation_token}, f)
            
            if pages_retrieved >= num_pages or not result.get('IsTruncated'):
                break

        return pdf_files

    # uploads single file to s3
    def upload_file(local_file_path, s3_key):
        s3.upload_file(local_file_path, bucket_name, s3_key)

    # uploads dir of files to s3
    def upload_directory_to_s3(ec2_dir, s3_dir, max_workers=48):
        upload_tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for root, dirs, files in os.walk(ec2_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    s3_key = os.path.join(s3_dir, os.path.relpath(local_file_path, ec2_dir)).replace("\\", "/")
                    upload_tasks.append(executor.submit(upload_file, local_file_path, s3_key))

            for future in as_completed(upload_tasks):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error uploading: {e}")

    # processing the pdfs: running through embedding pipeline and uploading to s3
    def process_pdfs(pdf_files, processor):
        print("IN PROCESS_PDFS: ", pdf_files)
        start_time = time.time()

        # PROCESS PDFS HERE 
        one, two, three, four, five, six = processor.pdfs_to_embeddings(pdf_files=pdf_files)
        pipeline_times['first'] += one
        pipeline_times['second'] += two
        pipeline_times['third'] += three 
        pipeline_times['fourth'] += four
        pipeline_times['fifth'] += five
        pipeline_times['sixth'] += six

        end_time = time.time()
        duration = end_time - start_time
        if duration > 0:
            throughput = len(pdf_files) / duration
        else:
            throughput = 0
        
        time1 = time.time()
        # UPLOADING EMBEDDINGS, TXTS, IMAGES TO S3 HERE 
        upload_directory_to_s3(txt_directory, data_dir_s3 + 'txt')
        print("finished uploading txt")
        upload_directory_to_s3(image_directory, data_dir_s3 + 'img')
        print("finished uploading img")
        upload_directory_to_s3(img_extracted_dir, data_dir_s3 + 'img_extracted')
        print("finished uploading img extracted")
        upload_directory_to_s3(embeddings_directory, data_dir_s3 + 'embeddings')
        print("finished uploading embed")
        upload_directory_to_s3(img_embeddings_dir, data_dir_s3 + 'embeddings_img_pg')
        print("finished uploading embed img pg")
        upload_directory_to_s3(e_img_embed_dir, data_dir_s3 + 'embeddings_img_extracted')
        print("finished uploading embed img extracted")

        time2 = time.time()

        with open("seq_times.txt", "a") as f:
            f.write(f"uploading all files to s3: {time2 - time1}\n")

        pipeline_times['sixth'] += time2-time1
        print("finished uploading current batch")
        print("pipeline times: ", pipeline_times)


    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE, processor):
        # result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
        # # get list of pdf file names
        # pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000

        
        overall_start_time = time.time()

        # get the pdf files from s3
        time_list = time.time()
        pdf_files = get_n_pdfs(1)
        pipeline_times['list'] = time.time() - time_list

        print("Now starting with total number of PDF files: ", len(pdf_files))

        local_batch = []

        for i in range(0, len(pdf_files), BATCH_SIZE):
            print('*****************************************************************************************************')
            print("WE ARE ON BATCH: ", i)
            print('*****************************************************************************************************')
            batch = pdf_files[i:i + BATCH_SIZE] 
            local_batch = []
            time_download = time.time()
            for pdf in batch:
                file_name = os.path.basename(pdf)
                local_path = os.path.join(pdf_directory, file_name)  # save here?
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket_name, pdf, local_path)
                local_batch.append(file_name)
            pipeline_times['download'] += time.time() - time_download

            process_pdfs(local_batch, processor)

            # delete the directories 
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR)
                os.makedirs(DATA_DIR, exist_ok=True)

            if os.path.exists(pdf_directory):
                shutil.rmtree(pdf_directory)
                os.makedirs(pdf_directory, exist_ok=True)
        
        overall_end_time = time.time()

        print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))
        print("TOTAL TIME list pdfs:",  pipeline_times['list'])
        print("TOTAL TIME download pdfs:",  pipeline_times['download'])
        print("TOTAL TIME pdf -> txt time:",  pipeline_times['first'])
        print("TOTAL TIME txt -> embed time:", pipeline_times['second'])
        print("TOTAL TIME pdf -> img per page time:", pipeline_times['third'])
        print("TOTAL TIME img per page -> embed time:", pipeline_times['fourth'])
        print("TOTAL TIME img extracted -> embed time :", pipeline_times['fifth'])
        print("TOTAL TIME uploading data:", pipeline_times['sixth'])

    def main():
        batched_file_download(BATCH_SIZE, processor) 

    main()
    text_model.model.stop_multi_process_pool(model_pool)
