import requests
from warcio.archiveiterator import ArchiveIterator
import os
import boto3
import pyarrow as pa
import pyarrow.parquet as pq

s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')

def extract_pdfs_to_s3(bucket_name, prefix, output_bucket, output_prefix):
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    schema = pa.schema({
        ('pdf', pa.string()),
        ('warc_filename', pa.string())
    })
    writer = pq.ParquetWriter('pdf_index.parquet', schema)
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                
                print(key)
                response = s3.get_object(Bucket=bucket_name, Key=key)
                body = response['Body']
                save_pdf_to_s3(body, key, output_bucket, output_prefix, schema, writer)
    writer.close()

def save_pdf_to_s3(body, warc_file, output_bucket, output_prefix, schema, writer):
    for record in ArchiveIterator(body):
        if record.rec_type == 'response':
            content_type = record.http_headers.get_header('Content-Type') if record.http_headers else ''
            if content_type and 'application/pdf' in content_type.lower():
                uri = record.rec_headers.get_header('WARC-Target-URI') or ''
                digest = record.rec_headers.get_header('WARC-Payload-Digest')[5:]
                # if no digest, compute hash
                row = {
                    'pdf': digest,
                    'warc_filename': warc_file
                }
                table = pa.Table.from_pydict({k: [v] for k, v in row.items()}, schema=schema)
                writer.write_table(table)

                s3_key = f"{output_prefix}{digest + '.pdf'}"
                print(s3_key)
                file = record.content_stream().read()
                s3.put_object(
                    Bucket=output_bucket,
                    Key=s3_key,
                    Body=file,
                    ContentType='application/pdf'
                )


extract_pdfs_to_s3("eotarchive", "crawl-data/EOT-2020/segments/IA-019/warc/", "bcgl-public-bucket", "2020_EOT_PDFs/PDFs/")