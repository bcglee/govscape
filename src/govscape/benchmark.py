#pip install beir

from beir import util
from beir.datasets.data_loader import GenericDataLoader

dataset = "nfcorpus"  # Choose a dataset
data_path = f"datasets/{dataset}"  # Directory where data will be stored

# Download and load dataset
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
util.download_and_unzip(url, data_path)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
