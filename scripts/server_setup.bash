#!/usr/bin/env bash
set -e

# Server setup
poetry run python scripts/download_s3_embeddings.py --bucket bcgl-public-bucket --folder 2008_EOT_PDFs/data_test_100k_final/ --local-dir ./data --max-workers 8
poetry run python scripts/start_api_server.py --pdf-dir ./data/PDFs --data-dir ./data --model UAE

