# pip install pdfplumber 
# pip install torch
# pip install transformers 
# pip install numpy

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
from .pdf_to_jpeg import PdfToJpeg
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
import time
import math
import re
import logging

# NOTE FOR THIS VERSION: *******************************************************************************************************

# 1. pdf -> txt -> embed
# 2. pdf -> img/pg -> embed
# 3.  pdf -> extracted img -> embed (not yet implemented)

# handles given a list of pdfs instead of given a dir of pdfs 

# *************************************************************************************************************

# global vars
BATCH_SIZE = 32

class EmbeddingModel(ABC):
    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_image(self, jpg_path):
        pass

class TextEmbeddingModel(EmbeddingModel):
    def __init__(self):
        if torch.cuda.is_available():
            print("USING GPU")
        else:
            print("USING CPU")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("WhereIsAI/UAE-Large-V1").to(self.device)  # note: max length = 512 
        #self.model = SentenceTransformer("WhereIsAI/UAE-Small-V1", device=self.device)
        #self.model = SentenceTransformer('distilbert-base-nli-mean-tokens').to(self.device)
        self.d = 1024
        self.image_to_caption = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=0 if torch.cuda.is_available() else -1)

        # multi-gpu version: 
        # self.model = SentenceTransformer("WhereIsAI/UAE-Large-V1", use_fast=True).to(self.device)
        # self.pool = model.start_multi_process_pool()
    
    def encode_text(self, text):
        with torch.no_grad():
            embeddings = self.model.encode(text, batch_size=32, device=self.device) # hopefully in batches
        return embeddings
    
    def encode_text_batch(self, texts): # TODO: verify you can put in a list of text files to do this in batches
        with torch.no_grad():
            embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, device=self.device) # hopefully in batches
        return embeddings  # can only convert embeddings to numpy on cpu?? 
    
    # def encode_text_batch_gpus(self, texts):
    #     return self.model.encode_multi_process(texts, self.pool)

    def encode_image(self, jpg_path): # output: embed_shape 
        image = Image.open(jpg_path)

        with torch.no_grad():
            caption = (self.image_to_caption(image))[0]['generated_text']
        # print(caption)
        image_caption_embed = self.model.encode([caption])

        return image_caption_embed
    
    
class CLIPEmbeddingModel(EmbeddingModel):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.d = 512
    
    def encode_text(self, text):
        #tokenize text
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        tokenized_text = inputs['input_ids'][0]

        #CLIP token limit = 77 so we have to divide into chunks and get embeddings for each of those
        max_chunk_len = 77
        text_chunks = []
        for i in range(0,len(tokenized_text), max_chunk_len):
            if len(tokenized_text[i:i+max_chunk_len]) == max_chunk_len or len(text_chunks) == 0:
                text_chunks.append(tokenized_text[i:i+max_chunk_len])

        #stack them all into a single batch so we can compute them all at the same time
        chunk_tensors = [] 
        for chunk in text_chunks:
            chunk_tensors.append(chunk.unsqueeze(0))
        batch_input_ids = torch.cat(chunk_tensors, dim=0)  
        batch_attention_mask = torch.ones_like(batch_input_ids)

        with torch.no_grad():
            batch_embeddings = self.model.get_text_features(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        embeddings = batch_embeddings.split(1, dim=0) 

        #decision: average embedding to create one embedding per PDF 
        final_embedding = torch.mean(torch.stack(embeddings), dim=0).to("cpu").numpy()

        return final_embedding

    def encode_image(self, jpg_path):
        image = Image.open(jpg_path)

        # preprocess image
        inputs = self.processor(images = image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        return image_embedding


class PDFsToEmbeddings:
    def __init__(self, pdf_directory, txt_directory, embeddings_dir, jpgs_dir, embedding_model):
        self.pdfs_path = pdf_directory
        self.txts_path = txt_directory
        self.embeddings_path = embeddings_dir
        self.jpgs_path = jpgs_dir
        self.embedding_model = embedding_model

        # TODO: uncomment for metadata
        #big json file turn into dictionary
        # self.json_file = "/homes/gws/cgong16/govscape/data/test_data/TechnicalReport234PDFs" # input path here, probably want to add to a script later since i assume this will always be there
        # self.json = {}
        # self.convert_json_to_dict(self.json_file)
        # print("FINISHED WITH JSON DICTIONARY")

    # TODO: uncomment for metadata
    # converts json file of metadata to a dictionary where key = digest, value = (govname, timestamp)
    # def convert_json_to_dict(self, json_file):
    #     with open(json_file, 'r') as file:
    #         data = json.load(file)
    #     for row in data:
    #         govname = row['govname']
    #         timestamp = row['timestamp']
    #         digest = row['digest']
    #         self.json[digest] = (govname, timestamp)


    # *******************************************************************************************************************
    # this is the og dif pdf -> dir txt -> dir embed pipeline 
    # *******************************************************************************************************************

    # (1) pdf -> txt 

    # converts a single pdf file to a txt files (one txt per page)
    def convert_pdf_to_txt(self, pdf_file):
        # print("Paginating & Scraping Text: " + pdf_file)
        pdf_path = os.path.join(self.pdfs_path, pdf_file)
        # print(pdf_path)
        #subdir for each pdf 
        pdf_subdir = os.path.join(self.txts_path, os.path.splitext(pdf_file)[0])

        if os.path.exists(pdf_subdir):
            return

        self.ensure_dir(pdf_subdir)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = []
                for page in pdf.pages:
                    text.append(page.extract_text())
        except:
            text = ["",]
            
        for page_num, page_text in enumerate(text):
            txt_file_path = os.path.join(pdf_subdir, f'{os.path.splitext(pdf_file)[0]}_{page_num}.txt')
            if len(page_text) == 0:
                continue
            with open(txt_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(page_text)
    
    # converts dir of pdfs -> dir of subdirs of txt files of each page AKA OVERALL PDFS -> TXTS
    # 1. OG VERSION 
    # def convert_pdfs_to_txt(self, pdf_files):
    #     self.ensure_dir(self.txts_path)
    #     for pdf_file in pdf_files:
    #         self.convert_pdf_to_txt(pdf_file)

    # 2. MP VERSION (from kyle)
    def convert_pdfs_to_txt(self, pdf_files=None):
        self.ensure_dir(self.txts_path)
        if pdf_files is None:
            pdf_files = os.listdir(self.pdfs_path)
        ctx = get_context('spawn')
        with ctx.Pool(processes=os.cpu_count()) as pool:
            pool.map(self.convert_pdf_to_txt, pdf_files)

    # 3. MT VERSION 
    # def convert_pdfs_to_txt(self, pdf_files):
    #     self.ensure_dir(self.txts_path)
    #     with ThreadPoolExecutor() as executor:
    #         executor.map(self.convert_pdf_to_txt, pdf_files)

    # (2) txt -> embed

    # str -> embed
    def text_to_embeddings(self, text):
        return self.embedding_model.encode_text(text)

    def txt_to_text(self, text, txt_path):
        text = ""
        with open(txt_path, 'r') as file:
            text = file.read()
        return text 

    # single txt -> embed
    def convert_txt_to_embedding(self, txt_path):
        # need to convert .txt to text
        text = self.txt_to_text(txt_path)
        return self.embedding_model.encode_text(text)
    
    # TODO: uncomment for metadata
    # creates a json file specifying page_nums with file_name in the embedding_dir
    def create_json(self, page_nums, embedding_dir, file_name):
        # probably need to have an exception case where the pdf file name is not found in the self.json...
        data = {"num_pages" : page_nums, "gov_name" : self.json[file_name][0], "timestamp" : self.json[file_name][1]}
        json_file_path = os.path.join(embedding_dir, file_name + ".json")
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    
    # version 1 = single subdir: txt subdir -> embed subdir.
    def convert_subdir_to_embeddings(self, txt_subdir_path):
        #print("Embedding PDF: " + txt_subdir_path)
        #making the subdir that will hold the embeddings for each PDF 
        embed_name = os.path.basename(txt_subdir_path)
        embedding_dir = os.path.join(self.embeddings_path, embed_name)
        
        # If the subdir already exists, we assume that this step has already been done.
        if os.path.exists(embedding_dir):
            return

        self.ensure_dir(embedding_dir)

        #all txt files in the txt subdir input 
        txt_files = os.listdir(txt_subdir_path)

        # self.create_json(len(txt_files), embedding_dir, os.path.basename(embedding_dir))  # TODO: uncomment for metadata

        for txt_file in txt_files:
            txt_path = os.path.join(txt_subdir_path, txt_file)

            embedding = self.convert_txt_to_embedding(txt_path)

            file_name = os.path.splitext(txt_file)[0] + ".npy"
            output_path = os.path.join(embedding_dir, file_name)
            np.save(output_path, embedding)
    
    # multiple txt subdir paths -> multiple embed dirs
    # set so the number of page files matches the batch size. 
    def convert_subdirs_to_embeddings(self, txt_subdir_paths):
        text_batch = []
        corresponding_file_batch = []
        for txt_subdir_path in txt_subdir_paths:
            embed_name = os.path.basename(txt_subdir_path)
            embedding_dir = os.path.join(self.embeddings_path, embed_name)
            self.ensure_dir(embedding_dir)

            #all txt files in the txt subdir 
            txt_files = sorted(os.listdir(txt_subdir_path), key = natural_key)

            for txt_file in txt_files:
                txt_path = os.path.join(txt_subdir_path, txt_file)
                text = self.txt_to_text(txt_file)
                text_batch.append(text)
                corresponding_file_batch.append((txt_file, embedding_dir))
                if len(txt_batch) == BATCH_SIZE:
                    batch_embedding = self.convert_text_batch(text_batch)

                    for (txt_name, embed_dir_path), embedding in zip(file_batch, batch_embedding):
                        file_name = txt_name.replace('.txt', '.npy')
                        output_path = os.path.join(embedding_dir, file_name)
                        np.save(output_path, embedding)
                    
                    text_batch = []
                    corresponding_file_batch = []
        
        # don't forget remaining 
        if text_batch:
            batch_embedding = self.convert_text_batch(text_batch)

            for (txt_name, embed_dir_path), embedding in zip(file_batch, batch_embedding):
                        file_name = txt_name.replace('.txt', '.npy')
                        output_path = os.path.join(embedding_dir, file_name)
                        np.save(output_path, embedding)

    # version 2 = batch of subdirs: txt subdir -> embed subdir.
    # def convert_subdir_to_embeddings(self, txt_subdir_paths):
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
    
    # txts -> embeds overall dir
    # 1. OG VERSION 
    # def convert_txts_to_embeddings(self, pdf_files):
    #     self.ensure_dir(self.embeddings_path)
    #     for pdf_file in pdf_files:
    #         subdir = os.path.join(self.txts_path, os.path.splitext(pdf_file)[0])
    #         if os.path.isdir(subdir):
    #             self.convert_subdir_to_embeddings(subdir)

    # 2. MP VERSION with pdf_files
    def convert_txts_to_embeddings(self):
        self.ensure_dir(self.embeddings_path)

        txt_subdirs_paths = []
        for txt_subdir in os.scandir(self.txts_path):
            if txt_subdir.is_dir():
                txt_subdirs_paths.append(txt_subdir.path)
        
        # splitting into groups for each process:   # TODO: verify concept: difference between passing in txt_subdir_batches and txt_subdirs_paths
        batch_size = math.ceil(len(txt_subdirs_paths) / os.cpu_count())
        txt_subdir_batches = []
        for i in range(0, len(txt_subdirs_paths), batch_size):
            txt_subdir_batches.append(txt_subdirs_paths[i : i + batch_size])

        ctx = get_context('spawn')
        with ctx.Pool(processes=os.cpu_count()) as pool:
            pool.map(self.convert_subdirs_to_embeddings, txt_subdir_batches) # for batch
            # pool.map(self.convert_subdir_to_embeddings, txt_subdirs_paths) # not in batch i believe


    # *******************************************************************************************************************
    # 1. this is the dir pdf -> dir img (of entire page) -> dir embed (of entire page) shared with og embed dir
    # *******************************************************************************************************************

    # pdfs -> dir of subdirs of extracted images
    # version1: entire path 
    # def convert_pdfs_to_single_jpg(self):
    #     if not os.path.exists(self.jpgs_path):
    #         os.makedirs(self.jpgs_path)
    #
    #     parser = PdfToJpeg(self.pdfs_path, self.jpgs_path, 100)
    #     parser.convert_directory_to_jpegs()
    
    # version2: given a list of files
    def convert_pdfs_to_single_jpg(self, pdf_files):
        parser = PdfToJpeg(self.pdfs_path, self.jpgs_path, 100)
        parser.convert_directory_to_jpegs(pdf_files)
    
    # img subdir -> embeds
    def convert_img_subdir_to_embeddings(self, jpg_subdir_path):
        print("Embedding PDF Img: " + jpg_subdir_path)
        #making the subdir that will hold the embeddings for each PDF 
        embed_name = os.path.basename(jpg_subdir_path)
        embedding_dir = os.path.join(self.embeddings_path, embed_name)

        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        # all jpg files in the jpg subdir input 
        jpg_files = os.listdir(jpg_subdir_path)

        for jpg_file in jpg_files:
            jpg_path = os.path.join(jpg_subdir_path, jpg_file)
            
            #check if file is empty; just skip to next if true 
            if os.stat(jpg_path).st_size == 0:
                continue
            try:
                embedding = self.embedding_model.encode_image(jpg_path)
            except: 
                continue
            file_name = os.path.splitext(jpg_file)[0] + "_img.npy"
            output_path = os.path.join(embedding_dir, file_name)
            np.save(output_path, embedding)


    # dir of imgs/pg -> dir of embeds
    # 1. OG VERSION 
    # def convert_imgs_to_embeddings(self):
    #     if not os.path.exists(self.embeddings_path):
    #         os.makedirs(self.embeddings_path)

    #     for jpg_subdir in os.scandir(self.jpgs_path):
    #         if jpg_subdir.is_dir():
    #             self.convert_img_subdir_to_embeddings(jpg_subdir.path)
    # 2. MP VERSION
    def convert_imgs_to_embeddings(self):
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)

        jpg_subdirs_paths = []
        for jpg_subdir in os.scandir(self.jpgs_path):
            if jpg_subdir.is_dir():
                jpg_subdirs_paths.append(jpg_subdir.path)
        
        batch_size = math.ceil(len(jpg_subdirs_paths) / os.cpu_count())
        jpg_subdir_batches = []
        for i in range(0, len(jpg_subdirs_paths), batch_size):
            jpg_subdir_batches.append(jpg_subdirs_paths[i : i + batch_size])

        ctx = get_context('spawn')
        with ctx.Pool(processes=os.cpu_count()) as pool:
            pool.map(self.convert_img_subdir_to_embeddings, jpg_subdirs_paths)
            # pool.map(self.convert_img_subdir_to_embeddings, jpg_subdir_batches)  #TODO: uncomment and figure out
    
    # *******************************************************************************************************************
    # dir pdf --> dir img (extracted) -> dir embed (extracted) shared with og embed dir
    # *******************************************************************************************************************

    # single pdf -> extracted img, extracted img embedding (using og embed dir)
    def extract_img_embed_pdf(self, pdf_path, output_img_dir_path, out_embed_path):
        pdf_doc = fitz.open(pdf_path)

        title = os.path.splitext(os.path.basename(pdf_path))[0]

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            for i, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                image_dict = pdf_doc.extract_image(xref)
                image_bytes = image_dict["image"]
                image = Image.open(io.BytesIO(image_bytes))

                image_path = Path(output_img_dir_path) / f"{title}_{page_num}_IMG_{i}.jpg"
                # print("img saved at: ",  image_path)
                image.save(image_path, "JPEG")

                # convert to embedding 
                embed = self.embedding_model.encode_image(image_path)

                output_path = os.path.join(out_embed_path, f"{title}_{page_num}_IMG_{i}.npy")
                np.save(output_path, embed)

    # pdfs -> extracted imgs, extracted img embeds
    def extract_img_pdfs(self):
        # go through entire set of pdfs 
        pdfs_dir = Path(self.pdfs_path)
        pdf_paths = list(pdfs_dir.glob("*.pdf"))

        extract_folder = Path(str(self.jpgs_path) + "_extract")
        extract_folder.mkdir(parents=True, exist_ok=True)

        # extract images and put it in images under _IMG_count.png
        for pdf_path in pdf_paths:
            img_path = Path((self.jpgs_path + "_extract")) / Path(pdf_path.stem)
            img_path.mkdir(parents=True, exist_ok=True)
            out_embed_path = Path(self.embeddings_path) / Path(pdf_path.stem)
            self.extract_img_embed_pdf(pdf_path, img_path, out_embed_path)
    
    # *******************************************************************************************************************
    # overall pipeline
    # *******************************************************************************************************************

    # version1: entire path
    # def pdfs_to_embeddings(self):
    #     self.convert_pdfs_to_txt()
    #     self.convert_txts_to_embeddings()
    #     # self.convert_pdfs_to_single_jpg()
    #     #self.convert_imgs_to_embeddings()  # i think this is for img of pdf -> embeddings ??
    #     #self.extract_img_pdfs() # add param to embed pipeline to use  # i think this is for extract img -> caption -> embeddings

    # version2: by list of pdf_files
    def pdfs_to_embeddings(self, pdf_files=None):
        pdf_files = pdf_files or os.listdir(self.pdfs_path)
        time1 = time.time()
        self.convert_pdfs_to_txt(pdf_files)
        time2 = time.time()
        self.convert_txts_to_embeddings()
        time3 = time.time()
        self.convert_pdfs_to_single_jpg(pdf_files)  # getting entire pdf page as an image. 
        time4 = time.time()
        self.convert_imgs_to_embeddings()  # image of entire pdf page (for document type in future)
        time5 = time.time()
        # self.extract_img_pdfs()  # extracted images and their embeddings #TODO: figure out this later + speed 

        first = time2 - time1
        sec = time3 - time2
        third = time4 - time3
        fourth = time5 - time4

        print("pdf -> txt time: ", time2 - time1)
        print("txt -> embed time: ", time3 - time2)
        print("pdf -> img per page time: ", time4 - time3)
        print("img per page -> embed time: ", time5 - time4)

        return first, sec, third, fourth

    # *******************************************************************************************************************
    # helper functions
    # *******************************************************************************************************************

    # makes sure that the directory specified is created
    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    # for sorting file names with page numbers to ensure consistency when batching between txt and npy files (OS could 
    # order file names differently)
    def natural_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

