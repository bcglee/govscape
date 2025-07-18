#!/usr/bin/env bash
set -e

# Server setup
s5cmd sync 's3://bcgl-public-bucket/prod-serving/*' data/prod
#poetry run python scripts/download_s3_embeddings.py --bucket bcgl-public-bucket  --prefix prod-serving --local-dir ./data/prod --max-workers 8
poetry run python scripts/start_api_server.py --pdf-directory ./data/PDFs --data-directory ./data/prod --text_model SentenceTransformer --visual_model CLIP 

