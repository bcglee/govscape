poetry run python3 scripts/data_prep/process_cdxs.py \
  --backend s3 \
  --bucket_name 'bcgl-public-bucket' \
  --output_prefix 'archive/CDX'

#poetry run python3 scripts/data_prep/retrieve_pdfs.py \
#  --backend s3 \
#  --bucket_name 'bcgl-public-bucket' \
#  --cdx_parquet 'archive/CDX/complete_cdx.parquet' \
#  --output_dir 'archive/PDFs'
