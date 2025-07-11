#!/usr/bin/env bash
set -e

# Embeddings pipeline
poetry run python s3_ec2_embedding_pipeline.py
