# pip install pdfplumber 
# pip install torch
# pip install transformers 
# pip install numpy

# for pdf -> txt
import pdfplumber 
import os
# for pdf -> jpeg
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
import fitz
# for CLIP
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool, TimeoutError, get_context

#for saving embeddings
import numpy as np
#abstract method
from abc import ABC, abstractmethod
import json
import sys
from .pdf_to_jpeg import PdfToJpeg

# 1. extract text of PDF files -> and outputs them to .txt files
#   - has stucture: dir -> subdir for each PDF -> .txt files of each page 
# 2. Plug in .txt files to CLIP to generate embeddings -> save as .npy file (could implement .pt if needed)
#   - has structure: dir -> subdir for each PDF -> .npy files for each page embedding 
# Note: this version supports creating a json metadata file for each pdf. So far it contains number of pages.

class EmbeddingModel(ABC):
    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_image(self, jpg_path):
        pass

class TextEmbeddingModel(EmbeddingModel):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("WhereIsAI/UAE-Large-V1").to(self.device)
        self.d = 1024
    
    def encode_text(self, text):
        #tokenize text
        text_embedding = self.model.encode([text])
        return text_embedding

    def encode_image(self, jpg_path):
        image = Image.open(jpg_path)

        # preprocess image
        inputs = self.processor(images = image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        return np.zeros(1024)
    

    
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

    #1. PDF -> TXT 

    # converts a single pdf file to a txt file
    def convert_pdf_to_txt(self, pdf_file):
        print("Paginating & Scraping Text: " + pdf_file)
        pdf_path = os.path.join(self.pdfs_path, pdf_file)
        #subdir for each pdf 
        pdf_subdir = os.path.join(self.txts_path, os.path.splitext(pdf_file)[0])

        # If the subdir already exists, we assume that this step has already been done.
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

    # # extracts the images from each pdf and saves as a jpeg
    # def convert_pdfs_to_jpg(self):
    #     if not os.path.exists(self.jpgs_path):
    #         os.makedirs(self.jpgs_path)
        
    #     pdf_files = os.listdir(self.pdfs_path)

    #     for pdf_file in pdf_files:
    #         pdf_path = os.path.join(self.pdfs_path, pdf_file)
            
    #         #subdir for each pdf 
    #         pdf_subdir = os.path.join(self.jpgs_path, os.path.splitext(pdf_file)[0])
    #         if not os.path.exists(pdf_subdir):
    #             os.makedirs(pdf_subdir)
            
    #         pdf = fitz.open(pdf_path)

    #         for page_num, page in enumerate(pdf):
    #             images = page.get_images(full=True)
    #             for img_ind, img in enumerate(images):
    #                     # way to reference the image
    #                     img_ref = img[0]
    #                     base_img = pdf.extract_image(img_ref)
    #                     img_bytes = base_img["image"]

    #                     new_jpg = Image.open(BytesIO(img_bytes))

    #                     img_file_path = os.path.join(pdf_subdir, f'{os.path.splitext(pdf_file)[0]}_{page_num}_{img_ind}.jpg')
    #                     new_jpg.save(img_file_path, "JPEG")
    
    # converts each pdf to an image
    def convert_pdfs_to_single_jpg(self):
        if not os.path.exists(self.jpgs_path):
            os.makedirs(self.jpgs_path)
        
        parser = PdfToJpeg(self.pdfs_path, self.jpgs_path, 100)
        parser.convert_directory_to_jpegs()

    # converts a dir of pdfs to a dir of subdirs for each pdf of txt files of each page 
    def convert_pdfs_to_txt(self):
        self.ensure_dir(self.txts_path)
        
        pdf_files = os.listdir(self.pdfs_path)
        ctx = get_context('spawn')
        with ctx.Pool(processes=12) as pool:
            pool.map(self.convert_pdf_to_txt, pdf_files)

    #2. TXT -> CLIP EMBEDDINGS

    #takes in a string and converts it into an embedding. 
    def text_to_embeddings(self, text):
        return self.embedding_model.encode_text(text)

    # takes in a single txt file and converts it into an embedding
    # added in some batch processing to make it faster
    def convert_txt_to_embedding(self, txt_path):
        # need to convert .txt to text
        with open(txt_path, 'r') as file:
            text = file.read()
        return self.embedding_model.encode_text(text)

    # creates a json file specifying page_nums in the embedding_dir with file_name
    def json_page_nums(self, page_nums, embedding_dir, file_name):
        data = {"page_nums" : page_nums}
        json_file_path = os.path.join(embedding_dir, file_name + ".json")
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    
    def convert_subdir_to_embeddings(self, txt_subdir_path):
        print("Embedding PDF: " + txt_subdir_path)
        #making the subdir that will hold the embeddings for each PDF 
        embed_name = os.path.basename(txt_subdir_path)
        embedding_dir = os.path.join(self.embeddings_path, embed_name)
        
        # If the subdir already exists, we assume that this step has already been done.
        if os.path.exists(embedding_dir):
            return

        self.ensure_dir(embedding_dir)

        #all txt files in the txt subdir input 
        txt_files = os.listdir(txt_subdir_path)

        self.json_page_nums(len(txt_files), embedding_dir, os.path.basename(embedding_dir))

        for txt_file in txt_files:
            txt_path = os.path.join(txt_subdir_path, txt_file)

            embedding = self.convert_txt_to_embedding(txt_path)

            file_name = os.path.splitext(txt_file)[0] + ".npy"
            output_path = os.path.join(embedding_dir, file_name)
            np.save(output_path, embedding)

    # converts a dir of subdirs for each pdf of txts for each page into a dir of subdir of embeddings in .npy
    def convert_txts_to_embeddings(self):
        self.ensure_dir(self.embeddings_path)

        txt_subdirs_paths = []
        for txt_subdir in os.scandir(self.txts_path):
            if txt_subdir.is_dir():
                txt_subdirs_paths.append(txt_subdir.path)
        
        ctx = get_context('spawn')
        with ctx.Pool(processes=2) as pool:
            pool.map(self.convert_subdir_to_embeddings, txt_subdirs_paths)

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
            np.save(output_path, embedding.cpu().numpy())

    # converts a dir of subdirs for each image for each page into a dir of subdir of embeddings in .npy
    def convert_imgs_to_embeddings(self):
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)

        jpg_subdirs_paths = []
        for jpg_subdir in os.scandir(self.jpgs_path):
            if jpg_subdir.is_dir():
                jpg_subdirs_paths.append(jpg_subdir.path)
        
        ctx = get_context('spawn')
        with ctx.Pool(processes=12) as pool:
            pool.map(self.convert_img_subdir_to_embeddings, jpg_subdirs_paths)


     # 1 + 2
    #converts a dir of pdfs to a dir of embeddings of .npy
    def pdfs_to_embeddings(self):
        self.convert_pdfs_to_txt()
        self.convert_txts_to_embeddings()
        self.convert_pdfs_to_single_jpg()
        self.convert_imgs_to_embeddings()

    #helper functions
    #makes sure that the directory specified is created
    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)