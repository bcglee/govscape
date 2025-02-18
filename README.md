# govscape
Searching millions of .gov PDFs


To run the initial version, you first build the embeddings, indices, etc. with:
```
python3 scripts/run_embedding_pipeline.py -p "data/test_data/TechnicalReport234PDFs" -d "data/test_data"
```

Then, you run the server with:
```
python3 scripts/start_server.py -p "data/test_data/TechnicalReport234PDFs" -d "data/test_data"
```
