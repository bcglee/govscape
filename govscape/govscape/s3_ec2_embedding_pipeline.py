import boto3
import os
import argparse
import time
import subprocess
import govscape as gs

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

def process_pdfs(pdf_files):
    start_time = time.time()
    print(f"Processing {len(pdf_files)} number of PDFs...")

    # Process the PDFs
    processor.pdfs_to_embeddings(pdf_files=pdf_files)
    
    end_time = time.time()
    duration = end_time - start_time
    if duration > 0:
        tp = len(pdf_files) / duration
    else:
        tp = 0
    
    print(f"Throughput for this batch: {tp} PDFs/sec (Processed {len(pdf_files)} PDFs in {duration} seconds)")

    upload_directory_to_s3(txt_directory, data_dir_s3 + 'txt')
    upload_directory_to_s3(embeddings_directory, data_dir_s3 + 'embeddings')
    print("Finished uploading.")
    # upload_directory_to_s3(image_directory, data_dir_s3 + 'images')

def batched_file_download(BATCH_SIZE):
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
    # Get list of pdf file names
    pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000

    print("Now starting with total number of PDF files: ", len(pdf_files))
    
    total_start_time = time.time()  # Start tracking total time

    total_gpu_utilization = 0
    total_memory_used = 0
    total_memory_free = 0
    total_pdf_count = 0

    # Process file batch by batch
    for i in range(0, len(pdf_files), BATCH_SIZE):
        batch = pdf_files[i:i + BATCH_SIZE] 

        # Download each file in the batch
        for pdf in batch:
            file_name = os.path.basename(pdf)

            local_path = os.path.join('downloads', file_name)  # Save to 'downloads' folder
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file into instance folder 'downloads'
            s3.download_file(bucket_name, pdf, local_path)
        
        process_pdfs(batch)

        # Capture GPU utilization after processing the batch
        gpu_utilization, memory_used, memory_free = get_gpu_utilization()
        if gpu_utilization is not None:
            total_gpu_utilization += float(gpu_utilization)
            total_memory_used += int(memory_used)
            total_memory_free += int(memory_free)
        
        total_pdf_count += len(batch)

    total_end_time = time.time()  # End tracking total time
    total_duration = total_end_time - total_start_time

    # Calculate overall throughput
    if total_duration > 0:
        total_throughput = total_pdf_count / total_duration
    else:
        total_throughput = 0

    # Average GPU utilization
    avg_gpu_utilization = total_gpu_utilization / len(pdf_files) if len(pdf_files) > 0 else 0
    avg_memory_used = total_memory_used / len(pdf_files) if len(pdf_files) > 0 else 0
    avg_memory_free = total_memory_free / len(pdf_files) if len(pdf_files) > 0 else 0

    # Output final throughput and GPU statistics
    print(f"Total time for processing: {total_duration} seconds")
    print(f"Total PDFs processed: {total_pdf_count}")
    print(f"Total throughput: {total_throughput} PDFs/sec")
    print(f"Average GPU utilization: {avg_gpu_utilization}%")
    print(f"Average memory used: {avg_memory_used}MB | Average memory free: {avg_memory_free}MB")

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
