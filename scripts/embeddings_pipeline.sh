#!/usr/bin/env bash
set -e

# Embeddings pipeline
rm -rf data/prod
poetry run python scripts/s3_ec2_embedding_pipeline.py --num_pages_to_process 10

