# govscape
Searching millions of .gov PDFs

## Build

To build the govscape server, we use the poetry build system. If poetry is properly installed, then running the following command should build the package:

```
poetry install
```

## Run

To run the initial version, you first build the embeddings, indices, etc. with:
```
poetry run python3 scripts/run_embedding_pipeline.py -p "data/test_data/TechnicalReport234PDFs" -d "data/test_data"
```

Then, you run the server with:
```
poetry run python3 scripts/start_server.py -p "data/test_data/TechnicalReport234PDFs" -d "data/test_data"
```
