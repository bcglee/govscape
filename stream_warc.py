import requests
from warcio.archiveiterator import ArchiveIterator
import os
import boto3
import pyarrow as pa
import pyarrow.parquet as pq

# extra
def extract_pdf_from_warc(pdf, warc_file_path, output_dir):
    # parquet file of pdf name and warc file retrived from
    
    with open(warc_file_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                uri = record.rec_headers.get_header('WARC-Target-URI') or ''
                # if uri and uri.lower().endswith('.pdf') and pdf in uri:
                if uri and pdf in uri:
                    content_type = record.http_headers.get_header('Content-Type') if record.http_headers else ''
                    if content_type and 'application/pdf' in content_type.lower():
                        if not pdf.lower().endswith('.pdf'):
                            pdf += '.pdf'
                        output_pdf = os.path.join(output_dir, pdf)
                        # try:
                        #     pdf_data = record.content_stream().read()

                        #     # Quick sanity check: does the content look like a PDF?
                        #     if not pdf_data.startswith(b'%PDF'):
                        #         print(f"⚠️ Skipping invalid PDF: {uri}")
                        #         continue

                        #     with open(output_pdf, 'wb') as file:
                        #         file.write(pdf_data)

                        # except Exception as e:
                        #     print(f"❌ Failed to write PDF from {uri}: {e}")
                        #     continue
                        with open(output_pdf, 'wb') as file:
                            file.write(record.content_stream().read())
                            print(f"PDF saved to: {output_pdf}")
                            return output_pdf
    print(f"PDF named {pdf} not found in WARC file")
    return None

def pdf_from_s3_warc_paths(warc_paths, output_dir):
    s3 = boto3.client('s3')
    bucket_name = 'eotarchive'
    prefix = 'crawl-data/EOT-2020/segments/IA-019/warc/'
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    i = 0
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                if i < 1:
                    key = obj['Key']
                    # print(key)
                    
                    # Get the object and read its content
                    response = s3.get_object(Bucket=bucket_name, Key=key)
                    body = response['Body']
                    extract_pdfs_from_warc(body, output_dir)
                    i = i + 1
                else:
                    break
                

    # with open(warc_paths, 'rb') as file:
    #     for i, line in enumerate(file):
    #         if (i < 2):
    #             # print(line.decode('utf-8').strip())
    #             extract_pdfs_from_warc(line.decode('utf-8').strip(), output_dir)
    #         else:
    #             break

def extract_pdfs_from_warc(body, output_dir):
    # with open(warc_file_path, 'rb') as stream:
        for record in ArchiveIterator(body):
            if record.rec_type == 'response':
                content_type = record.http_headers.get_header('Content-Type') if record.http_headers else ''
                if content_type and 'application/pdf' in content_type.lower():
                    uri = record.rec_headers.get_header('WARC-Target-URI') or ''
                    pdf = record.rec_headers.get_header('WARC-Payload-Digest')
                    if not pdf.lower().endswith('.pdf'):
                        pdf += '.pdf'
                    output_pdf = os.path.join(output_dir, pdf)

                    with open(output_pdf, 'wb') as file:
                        file.write(record.content_stream().read())
                        print(f"PDF saved to: {output_pdf}")
                    
def list_pdfs_in_warc(warc_file_path):
    pdf_uri = []
    pdf_base = []
    with open(warc_file_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                uri = record.rec_headers.get_header('WARC-Target-URI') or ''
                content_type = record.http_headers.get_header('Content-Type') if record.http_headers else ''
                if content_type and 'application/pdf' in content_type.lower():
                    pdf_uri.append(uri)
                    pdf_base.append(os.path.basename(uri))
    if not pdf_uri:
        print("No PDFs found in the WARC file.")
    print(pdf_uri)
    print(pdf_base)
    print(len(pdf_base))


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
# pdf_from_s3_warc_paths("warc.paths", "/homes/gws/alisony/govscape/hashed")
# list_pdfs_in_warc("EOT20-20210502114129-crawl865_EOT20-20210502114129-12224.warc.gz")
# extract_pdfs_from_warc("EOT20-20210502114129-crawl865_EOT20-20210502114129-12224.warc.gz", "/homes/gws/alisony/govscape/warc_output")
