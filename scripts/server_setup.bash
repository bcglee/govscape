#!/usr/bin/env bash
set -e

# Server setup
poetry run python download_s3_embeddings.py
poetry run python start_api_server.py
