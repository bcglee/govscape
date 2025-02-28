# govscape
Searching millions of .gov PDFs

## Build

To build the govscape server, we use the poetry build system. If poetry is properly installed, then running the following command should build the package:

```
poetry install
```

## Run

### Original Mode (Local PDFs)

1. First build the embeddings, indices, etc. with:

```
poetry run python3 scripts/run_embedding_pipeline.py -p "data/test_data/TechnicalReport234PDFs" -d "data/test_data"
```

2. Then, run the server with:

```
poetry run python3 scripts/start_server.py -p "data/test_data/TechnicalReport234PDFs" -d "data/test_data"
```

### Precomputed Mode

To run with precomputed embeddings downloaded from S3:

1. First download the embeddings from S3:

```
poetry run python3 scripts/download_s3_embeddings.py -o "data/downloaded_embeddings" --metadata
```

2. Then, run the server with the downloaded embeddings:

```
poetry run python3 scripts/start_server.py --mode s3-downloaded -e "data/downloaded_embeddings" -i "data/images"
```

## Command Line Options

### run_embedding_pipeline.py

```
-p, --pdf-directory: Directory containing PDFs to embed
-d, --data-directory: Directory to store embeddings and metadata
-v, --verbose: Enable verbose output
```

### start_server.py

```
--mode: Server mode (original or s3-downloaded), default is original
-p, --pdf-directory: Directory containing PDFs (original mode)
-d, --data-directory: Directory containing embeddings and metadata (original mode)
-e, --embeddings-dir: Directory containing downloaded embeddings (s3-downloaded mode)
-i, --image-dir: Directory containing JPEG images (s3-downloaded mode)
-k, --top-k: Number of results to return (default: 5)
-v, --verbose: Enable verbose output
```

### download_s3_embeddings.py

```
-o, --output-directory: Directory to save downloaded embeddings
-b, --batch-size: Batch size for downloading (default: 1000)
--metadata: Also download metadata files (text content)
```