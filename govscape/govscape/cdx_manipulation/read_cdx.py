import duckdb
import boto3
import pandas
import os 

def list_cdx_parquet_files(s3_bucket='eotarchive', s3_cdx_dir='eot-index/table/eot-main/crawl=EOT-2020/'):
    s3 = boto3.client('s3')

    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_cdx_dir)
    s3_cdx_keys = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        s3_cdx_keys.append(key)

    return s3_cdx_keys

def download_cdx_parquet(s3_bucket='eotarchive', s3_cdx_key='eot-index/table/eot-main/crawl=EOT-2020/', local_dir='/tmp/govscape'):
    s3 = boto3.client('s3')
    local_file_path = os.path.join(local_dir, os.path.basename(s3_cdx_key))
    return s3.download_file(s3_bucket, s3_cdx_key, local_file_path)

def compute_single_cdx_counts(local_cdx_filepath, only_pdfs=True):
    if only_pdfs:
        return duckdb.sql("Select regexp_extract(url, 'https?://([^//])*\\.gov') as domain, count(*) as domain_count FROM '" + local_cdx_filepath + "' where url LIKE '%.pdf' GROUP BY domain order by domain_count desc").df()
    else:
        return duckdb.sql("Select regexp_extract(url, 'https?://([^//])*\\.gov') as domain, count(*) as domain_count FROM '" + local_cdx_filepath + "' GROUP BY domain order by domain_count desc").df()