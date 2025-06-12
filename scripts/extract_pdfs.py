import requests
from warcio.archiveiterator import ArchiveIterator
import os
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import time
import multiprocessing


def save_pdf_to_s3(s3, body, warc_file, output_bucket, output_prefix, counter):
    for record in ArchiveIterator(body):
        if record.rec_type == 'response':
            content_type = record.http_headers.get_header('Content-Type') if record.http_headers else ''
            if content_type and 'application/pdf' in content_type.lower():
                uri = record.rec_headers.get_header('WARC-Target-URI') or ''
                digest = record.rec_headers.get_header('WARC-Payload-Digest')[5:]
                # if no digest, compute hash

                s3_key = f"{output_prefix}{digest + '.pdf'}"
                file = record.content_stream().read()
                start = time.time()
                s3.put_object(
                    Bucket=output_bucket,
                    Key=s3_key,
                    Body=file,
                    ContentType='application/pdf'
                )

                counter += 1
                print("URI " + uri + " Took " + str(time.time()-start) + " Seconds")
                print("Processed " + str(counter) + " PDFs")
                row = {
                    'pdf': digest,
                    'url': uri,
                    'warc_filename': warc_file
                }
                return row

def handle_single_warc(obj, bucket_name, output_bucket, output_prefix, counter):
    key = obj['Key']
    
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=key)
    body = response['Body']
    save_pdf_to_s3(s3, body, key, output_bucket, output_prefix, counter)

def extract_pdfs_to_s3(bucket_name, prefix, output_bucket, output_prefix):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    schema = pa.schema({
        ('pdf', pa.string()),
        ('warc_filename', pa.string())
    })
    writer = pq.ParquetWriter('pdf_index.parquet', schema)
    counter = 0
    for page in page_iterator:
        if 'Contents' in page:
            with multiprocessing.Pool(processes=4) as pool:
                row_sets = pool.starmap(handle_single_warc, [[page, bucket_name, output_bucket, output_prefix, counter] for page in page['Contents']])
            for row_set in row_sets:
                for rows in row_set:
                    for row in rows:
                        table = pa.Table.from_pydict({k: [v] for k, v in row.items()}, schema=schema)
                        writer.write_table(table)

    writer.close()


extract_pdfs_to_s3("eotarchive", "crawl-data/EOT-2020/segments/IA-019/warc/", "bcgl-public-bucket", "2020_EOT_PDFs/PDFs/")