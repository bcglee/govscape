import pdfplumber 
import os
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
import fitz
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, LoggingHandler
import torch
import torch.nn.functional as F
from transformers import pipeline
from pathlib import Path
import io
import numpy as np
from abc import ABC, abstractmethod
import json
import sys
# from .pdf_to_jpeg import PdfToJpeg
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
import time
import math
import re
import logging
import pynvml

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

# global vars
GPU_BATCH_SIZE = 2
BATCH_SIZE = 64

class EmbeddingModel(ABC):
    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_image(self, jpg_path):
        pass

def get_least_used_cuda():
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        min_used_mem = float("inf")
        best_device = "cuda:0"
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if meminfo.used < min_used_mem:
                min_used_mem = meminfo.used
                best_device = f"cuda:{i}"
        pynvml.nvmlShutdown()
        return best_device

class TextEmbeddingModel(EmbeddingModel):
    def __init__(self):
        if torch.cuda.is_available():
            print("USING GPU")
        else:
            print("USING CPU")
        self.device = get_least_used_cuda() if torch.cuda.is_available() else "cpu"
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.model = SentenceTransformer("WhereIsAI/UAE-Large-V1").to(self.device)  # note: max length = 512   #for querying hugging face
        self.model = SentenceTransformer('./uae-large-v1').to(self.device)  # for local
        #self.model = SentenceTransformer("WhereIsAI/UAE-Small-V1", device=self.device)
        #self.model = SentenceTransformer('distilbert-base-nli-mean-tokens').to(self.device)
        self.d = 1024
        # self.image_to_caption = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=0 if torch.cuda.is_available() else -1)

        # multi-gpu version: 
        # self.pool = self.model.start_multi_process_pool()

    def encode_text(self, text):
        with torch.no_grad():
            embeddings = self.model.encode(text, batch_size=GPU_BATCH_SIZE, device=self.device) # hopefully in batches
        return embeddings
    
    def encode_text_batch(self, texts): # TODO: verify you can put in a list of text files to do this in batches
        with torch.no_grad():
            embeddings = self.model.encode(texts, batch_size=GPU_BATCH_SIZE, device=self.device) # hopefully in batches
        return embeddings  # can only convert embeddings to numpy on cpu?? 
    
    # def encode_text_batch_gpus(self, texts):
    #     return self.model.encode_multi_process(texts, batch_size=GPU_BATCH_SIZE, self.pool)  

    def encode_image(self, jpg_path): # output: embed_shape    #TODO: use dataset so it doesn't process sequentially?? 
        image = Image.open(jpg_path)

        with torch.no_grad():
            caption = (self.image_to_caption(image))[0]['generated_text']
        # print(caption)
        image_caption_embed = self.model.encode([caption])

        return image_caption_embed
    

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

class TxtsToEmbeddings:
    def __init__(self, txt_directory, embeddings_dir):
        self.txts_path = txt_directory
        self.embeddings_path = embeddings_dir
        # self.jpgs_path = jpgs_dir
        # self.embedding_model = embedding_model


    # (2) txt -> embed

    def txt_to_text(self, txt_path):
        text = ""
        with open(txt_path, 'r') as file:
            text = file.read()
        return text 

    # single txt -> embed
    # def convert_txt_to_embedding(self, txt_path):
    #     # need to convert .txt to text
    #     text = self.txt_to_text(txt_path)
    #     return self.embedding_model.encode_text(text)
    
    
    # version 1 = single subdir: txt subdir -> embed subdir.
    # def convert_subdir_to_embeddings(self, txt_subdir_path):
    #     #print("Embedding PDF: " + txt_subdir_path)
    #     #making the subdir that will hold the embeddings for each PDF 
    #     embed_name = os.path.basename(txt_subdir_path)
    #     embedding_dir = os.path.join(self.embeddings_path, embed_name)
        
    #     # If the subdir already exists, we assume that this step has already been done.
    #     if os.path.exists(embedding_dir):
    #         return

    #     self.ensure_dir(embedding_dir)

    #     #all txt files in the txt subdir input 
    #     txt_files = os.listdir(txt_subdir_path)

    #     # self.create_json(len(txt_files), embedding_dir, os.path.basename(embedding_dir))  # TODO: uncomment for metadata

    #     for txt_file in txt_files:
    #         txt_path = os.path.join(txt_subdir_path, txt_file)

    #         embedding = self.convert_txt_to_embedding(txt_path)

    #         file_name = os.path.splitext(txt_file)[0] + ".npy"
    #         output_path = os.path.join(embedding_dir, file_name)
    #         np.save(output_path, embedding)

    # for sorting file names with page numbers to ensure consistency when batching between txt and npy files (OS could 
    # order file names differently)
    
    # multiple txt subdir paths -> multiple embed dirs
    # set so the number of page files matches the batch size. 
    def convert_subdirs_to_embeddings(self, txt_subdir_paths):
        text_batch = []
        file_batch = []
        # print(txt_subdir_paths)
        for txt_subdir_path in txt_subdir_paths:
            embed_name = os.path.basename(txt_subdir_path)
            embedding_dir = os.path.join(self.embeddings_path, embed_name)
            self.ensure_dir(embedding_dir)

            #all txt files in the txt subdir 
            # print("TXT SUBDIR PATH IS ", txt_subdir_path)
            # print(self.embeddings_path)
            # print(txt_subdir_path)
            txt_files = sorted(os.listdir(txt_subdir_path), key = natural_key)

            for txt_file in txt_files:
                txt_path = os.path.join(txt_subdir_path, txt_file)
                text = self.txt_to_text(txt_path)
                output_path = os.path.join(embedding_dir, txt_file)
                text_batch.append(text)
                file_batch.append(output_path)
        
        return text_batch, file_batch


    # 2. MP VERSION with pdf_files
    def convert_txts_to_embeddings(self):
        all_texts = []
        all_embed_file_paths = []
        self.ensure_dir(self.embeddings_path)

        txt_subdirs_paths = []
        for txt_subdir in os.scandir(self.txts_path):
            if txt_subdir.is_dir():
                txt_subdirs_paths.append(txt_subdir.path)
        
        # splitting into groups for each process:   # TODO: verify concept: difference between passing in txt_subdir_batches and txt_subdirs_paths
        batch_size = math.ceil(len(txt_subdirs_paths) / (os.cpu_count() // 2))
        # batch_size = math.ceil(len(txt_subdirs_paths) / 2)
        txt_subdir_batches = []
        for i in range(0, len(txt_subdirs_paths), batch_size):
            txt_subdir_batches.append(txt_subdirs_paths[i : i + batch_size])
        
        # print("txt_subdir_batches ", txt_subdir_batches)

        ctx = get_context('spawn')
        with ctx.Pool(processes=(os.cpu_count() // 2)) as pool:
        # with ctx.Pool(processes=(2)) as pool:
            results = pool.map(self.convert_subdirs_to_embeddings, txt_subdir_batches) # for batch
            # pool.map(self.convert_subdir_to_embeddings, txt_subdirs_paths) # not in batch i believe

            for text_batch, embed_file_path_batch in results:
                all_texts.extend(text_batch)
                all_embed_file_paths.extend(embed_file_path_batch)
        
        return all_texts, all_embed_file_paths
    
    # saves each embedding at the respective file
    def convert_embedding_to_files_batch(self, embed_and_paths):
        embed, embed_file_paths = embed_and_paths
        for output_path, embedding in zip(embed_file_paths, embed):
            file_name = output_path.replace('.txt', '.npy')
            # print(f"file_name: {file_name} has been saved.")
            np.save(file_name, embedding)
    
    # given an embedding, output each embedding into their respective embedding file paths
    def convert_embedding_to_files(self, embed, embed_file_paths):
        # split the embedding up into chunks
        # chunks = np.array_split(embed, (os.cpu_count() // 2))
        chunks = np.array_split(embed, 2)
        chunk_embed_file_paths = []

        start = 0
        for i in range(len(chunks)):
            end = chunks[i].shape[0]
            chunk_embed_file_paths.append(embed_file_paths[start: start + end])
            start = start + end 
        
        if len(chunks) != len(chunk_embed_file_paths):
            raise Exception("chunks and chunk_embed_file_paths should be the same length.")

        ctx = get_context('spawn')
        # with ctx.Pool(processes=(os.cpu_count() // 2)) as pool:
        with ctx.Pool(processes=(2)) as pool:
            pool.map(self.convert_embedding_to_files_batch, zip(chunks, chunk_embed_file_paths)) # for batch

    # *******************************************************************************************************************
    # helper functions
    # *******************************************************************************************************************

    # makes sure that the directory specified is created
    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

# def main(txt_path, embed_path):

#     processor = TxtsToEmbeddings(txt_path, embed_path)  # note, we are not using the model in here.
#     # print("txt_path is ", txt_path)

#     # sentences
#     sentences, all_embed_file_paths = processor.convert_txts_to_embeddings()  #txts to text

#     # Define the model
#     text_model = TextEmbeddingModel()
#     text_model.pool = text_model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

#     # Compute the embeddings using the multi-process pool
#     emb = text_model.model.encode_multi_process(sentences, text_model.pool)
#     print("Embeddings computed. Shape:", emb.shape)

#     # put them into embedding files 
#     processor.convert_embedding_to_files(emb, all_embed_file_paths)

#     # Optional: Stop the processes in the pool
#     text_model.model.stop_multi_process_pool(text_model.pool)

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    # print("HIHIHIIHIHIHI ********************************************************************************************************")
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    # print(PROJECT_ROOT)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'test_data')  # THIS IS WHERE THE OVERALL DATA DIR IS IN EC2
    txt_path = os.path.join(DATA_DIR, 'txt')
    embed_path = os.path.join(DATA_DIR, 'embeddings')
    # print(txt_path)
    # print(embed_path)

    processor = TxtsToEmbeddings(txt_path, embed_path)  # note, we are not using the model in here.
    # print("txt_path is ", txt_path)

    # sentences
    sentences, all_embed_file_paths = processor.convert_txts_to_embeddings()  #txts to text

    # Define the model
    text_model = TextEmbeddingModel()
    pool = text_model.model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

    # print("length of sentences is ", len(sentences))

    # Compute the embeddings using the multi-process pool
    emb = text_model.model.encode_multi_process(sentences, pool)
    print("Embeddings computed. Shape:", emb.shape)

    # put them into embedding files 
    processor.convert_embedding_to_files(emb, all_embed_file_paths)

    # Optional: Stop the processes in the pool
    text_model.model.stop_multi_process_pool(pool)

