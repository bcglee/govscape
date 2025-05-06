import boto3
import os
import argparse
import time
import subprocess
import govscape as gs
import concurrent.futures

s3 = boto3.client('s3')

# FIELDS TO SET **************************************************************************************

BATCH_SIZE = 10

# s3://bcgl-public-bucket/2008_EOT_PDFs/PDFs/
bucket_name = 'bcgl-public-bucket'
pdfs_dir = '2008_EOT_PDFs/PDFs/'
data_dir_s3 = '2008_EOT_PDFs/data2/' 
# data and data1 were for testing cpu file output
# data2 is for testing single gpu file output

# for processing pdfs: 
# pdf_directory = '../../data/test_data/small_test'
# txt_directory = '../../data/test_data/txt'
# embeddings_directory = '../../data/test_data/embeddings'
# image_directory = '../../data/test_data/images'

model = gs.TextEmbeddingModel()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'test_data')

#pdf_directory = os.path.join(DATA_DIR, 'small_test')  # for testing a folder of three pdfs
pdf_directory = os.path.join(DATA_DIR, 'TechnicalReport234PDFs')
txt_directory = os.path.join(DATA_DIR, 'txt')
embeddings_directory = os.path.join(DATA_DIR, 'embeddings')
image_directory = os.path.join(DATA_DIR, 'images')

processor = gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, image_directory, model)

# ****************************************************************************************************

def get_gpu_utilization():
    """Function to get the current GPU utilization using nvidia-smi."""
    try:
        # Run nvidia-smi to get GPU utilization
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        gpu_stats = result.stdout.strip().split(', ')
        gpu_utilization = gpu_stats[0]
        memory_used = gpu_stats[1]
        memory_free = gpu_stats[2]
        return gpu_utilization, memory_used, memory_free
    except subprocess.CalledProcessError as e:
        print(f"Error while getting GPU utilization: {e}")
        return None, None, None


def download_pdf(pdf, batch_download_dir):
    file_name = os.path.basename(pdf)
    local_path = os.path.join(batch_download_dir, file_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket_name, pdf, local_path)

def process_pdfs(pdf_files):
    start_time = time.time()
    print(f"Processing {len(pdf_files)} number of PDFs...")


    gpu_utilization_before, memory_used_before, memory_free_before = get_gpu_utilization()

    # Process the PDFs
    processor.pdfs_to_embeddings(pdf_files=pdf_files)

    gpu_utilization_after, memory_used_after, memory_free_after = get_gpu_utilization()
    
    end_time = time.time()
    duration = end_time - start_time
    throughput = len(pdf_files) / duration if duration > 0 else 0
    
    if gpu_utilization_before is not None and gpu_utilization_after is not None:
        gpu_utilization_diff = float(gpu_utilization_after) - float(gpu_utilization_before)
        memory_used_diff = int(memory_used_after) - int(memory_used_before)
        memory_free_diff = int(memory_free_after) - int(memory_free_before)
        print(f"GPU utilization change: {gpu_utilization_diff}%")
        print(f"Memory used change: {memory_used_diff}MB | Memory free change: {memory_free_diff}MB")
    
    print(f"Throughput for this batch: {throughput} PDFs/sec (Processed {len(pdf_files)} PDFs in {duration} seconds)")

    upload_directory_to_s3(txt_directory, data_dir_s3 + 'txt')
    upload_directory_to_s3(embeddings_directory, data_dir_s3 + 'embeddings')
    # upload_directory_to_s3(image_directory, data_dir_s3 + 'images')
    print("Finished uploading.")

# def batched_file_download(BATCH_SIZE):
#     result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
#     # Get list of pdf file names
#     pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000

#     # temporary: 
#     pdf_files = pdf_files[0:100]

#     print("Now starting with total number of PDF files: ", len(pdf_files))
    
#     total_start_time = time.time()

#     total_gpu_utilization = 0
#     total_memory_used = 0
#     total_memory_free = 0
#     total_pdf_count = 0

#     for i in range(0, len(pdf_files), BATCH_SIZE):
#         print("BATCH: ", i)
#         batch = pdf_files[i:i + BATCH_SIZE] 

#         for pdf in batch:
#             file_name = os.path.basename(pdf)
#             local_path = os.path.join('downloads', file_name)  # save here?
#             os.makedirs(os.path.dirname(local_path), exist_ok=True)
#             s3.download_file(bucket_name, pdf, local_path)
        
#         process_pdfs(batch)

#         gpu_utilization, memory_used, memory_free = get_gpu_utilization()
#         if gpu_utilization is not None:
#             total_gpu_utilization += float(gpu_utilization)
#             total_memory_used += int(memory_used)
#             total_memory_free += int(memory_free)
        
#         total_pdf_count += len(batch)

#     total_end_time = time.time()
#     total_duration = total_end_time - total_start_time

#     if total_duration > 0:
#         total_throughput = total_pdf_count / total_duration
#     else:
#         total_throughput = 0

#     avg_gpu_utilization = total_gpu_utilization / len(pdf_files) if len(pdf_files) > 0 else 0
#     avg_memory_used = total_memory_used / len(pdf_files) if len(pdf_files) > 0 else 0
#     avg_memory_free = total_memory_free / len(pdf_files) if len(pdf_files) > 0 else 0

#     print(f"Total time for processing: {total_duration} seconds")
#     print(f"Total PDFs processed: {total_pdf_count}")
#     print(f"Total throughput: {total_throughput} PDFs/sec")
#     print(f"Average GPU utilization: {avg_gpu_utilization}%")
#     print(f"Average memory used: {avg_memory_used}MB | Average memory free: {avg_memory_free}MB")


def batched_file_download(BATCH_SIZE):
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
    pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]

    # temporary: 
    pdf_files = pdf_files[:10]

    print("Now starting with total number of PDF files: ", len(pdf_files))
    

    total_start_time = time.time()

    batch_download_dir = "downloads"  # Change this as necessary

    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for pdf in pdf_files:
            futures.append(executor.submit(download_pdf, pdf, batch_download_dir))

        # Wait for all downloads to finish
        concurrent.futures.wait(futures)

    # After download, process PDFs as usual
    for i in range(0, len(pdf_files), BATCH_SIZE):
        batch = pdf_files[i:i + BATCH_SIZE]
        process_pdfs(batch)

def upload_directory_to_s3(ec2_dir, s3_dir):
    for root, dirs, files in os.walk(ec2_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.join(s3_dir, os.path.relpath(local_file_path, ec2_dir)).replace("\\", "/")

            print(f"Uploading {local_file_path} to {s3_key}...")
            s3.upload_file(local_file_path, bucket_name, s3_key)


#poetry run python s3_ec2_embedding_pipeline.py
def main():
    batched_file_download(BATCH_SIZE)

if __name__ == '__main__':
    main()
