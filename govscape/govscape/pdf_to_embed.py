import pdfplumber 
import os
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import fitz
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizer
from sentence_transformers import LoggingHandler
import torch
from pathlib import Path
import io
import numpy as np
from abc import ABC, abstractmethod
import json
from multiprocessing import get_context
import time
import re
import logging
import shutil
import multiprocessing as mp
from .pdf_to_jpeg import PdfToJpeg
from .pdf_to_embed_multigpu import TextEmbeddingModel, compute_text_embeddings


# OVERALL NOTE FOR THIS VERSION: ********************************************************************************************
# 1. pdf -> txt -> embed.
# 2. pdf -> img/pg -> embed. this calls the pdf_to_jpeg file and i believe the image embedding model is in this file.
# 3. pdf -> extracted img -> embed

# note: handles given a list of pdfs instead of given a dir of pdfs which i am now realizing might not be necessary 
# since s3_ec2_embedding_pipeline deletes all directories after uploading all generated files to s3...

# note: text embedding model for multi-gpu use is in pdf_to_embed_multigpu.py
# *******************************************************************************************************************


# NOTE for setting up an ec2 instance *******************************************************************************
# sudo yum update -y
# sudo yum install git -y
# git clone https://github.com/bcglee/govscape.git
# curl -sSL https://install.python-poetry.org | python3 -

# curl -O https://bootstrap.pypa.io/get-pip.py
# python3 get-pip.py --user

# sudo yum groupinstall "Development Tools" -y
# sudo yum install gcc openssl-devel bzip2-devel libffi-devel xz-devel wget make -y

# cd /usr/src
# sudo wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
# sudo tar xzf Python-3.10.14.tgz
# cd Python-3.10.14
# sudo ./configure --enable-optimizations
# sudo make altinstall

# poetry install
# poetry add boto3  # these probably should be added into the poetry file
# poetry add nvidia-ml-py3

# sudo yum install -y poppler-utils # for pdf -> img conversion 

# aws configure

# # for activating gpu: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html i think it's a little different depending on ec2 instance type

# # nice to have: 
# sudo yum install tmux
# aws s3 ls s3://bcgl-public-bucket/2008_EOT_PDFs/data_test_100k_2/embeddings/ --recursive | awk -F'/' '{print $4}' | sort | uniq | wc -l  # ex. for checking how many are in a file

# *******************************************************************************************************************

# IMPORTANT NOTE!!! ********************************************************************************************
# this code can support using the model from hugging face online (currently commented out) and also local when bulk requests are limited. 

# to download the models locally: 
# pip install huggingface_hub
# huggingface-cli download WhereIsAI/UAE-Large-V1 --local-dir ./uae-large-v1  # text model 
# git clone https://huggingface.co/openai/clip-vit-base-patch32  # image model 
# git clone https://huggingface.co/nlpconnect/vit-gpt2-image-captioning  # optional, if wanting to put extracted images -> caption -> uae text model to be searched in the same space as text
# *******************************************************************************************************************


# global vars *******************************************************************************************************
GPU_BATCH_SIZE = 2
BATCH_SIZE = 64
# *******************************************************************************************************************

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

class EmbeddingModel(ABC):
    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_image(self, jpg_path):
        pass

class CLIPEmbeddingModel(EmbeddingModel):
    def __init__(self):
        self.d = 512

    def encode_text(self, text):  #note: not doing encode_texts version of this yet because currently not in use. 
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
        image = Image.open(jpg_path).convert("RGB")

        # preprocess image
        inputs = self.processor(images = image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        return image_embedding[0].to("cpu").numpy()
    
    # single gpu case 
    # def encode_images(self, jpg_paths):
        # batch_size = 32
        # images = []
        # for jpg_path in jpg_paths:
        #     img = Image.open(jpg_path)
        #     images.append(img)
        
        # inputs = self.processor(images=images, return_tensors="pt")
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
        # with torch.no_grad():
        #     if isinstance(self.model, torch.nn.DataParallel):  #multi-gpu case
        #         embeddings = self.model.module.get_image_features(**inputs)
        #     else:
        #         embeddings = self.model.get_image_features(**inputs)
        #     embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # return embeddings.cpu().numpy()
    
    # for multi-gpus: 

    def create_model_and_processor(self, gpu_id):
        device = torch.device(f"cuda:{gpu_id}")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # online
        model.eval()

        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)  # online
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")  # online
        #image_processor = CLIPImageProcessor.from_pretrained("./clip-vit-base-patch32", use_fast=True)  # local
        #tokenizer = CLIPTokenizer.from_pretrained("./clip-vit-base-patch32")  # local

        processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)

        return model, processor, device


    def encode_images_per_gpu(self, jpg_paths, gpu_id, max_batch_size, results):
        model, processor, device = self.create_model_and_processor(gpu_id)

        all_embeddings = []

        for i in range(0, len(jpg_paths), max_batch_size):  # idea: move this into a multiprocessing method and then pass in the images instead of jpg paths into the gpus. 
            batch_paths = jpg_paths[i:i + max_batch_size]
            images = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    if img.size[0] < 70 or img.size[1] < 70:
                        continue
                    images.append(img)
                except Exception as e:
                    continue

            if not images:
                continue

            try:
                inputs = processor(images=images, return_tensors="pt", input_data_format="channels_last")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    embeddings = model.get_image_features(**inputs)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                all_embeddings.append(embeddings.cpu())

            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

        if all_embeddings:
            results[gpu_id] = torch.cat(all_embeddings, dim=0).numpy()
        else:
            results[gpu_id] = np.empty((0, self.d), dtype=np.float32)
    
    def encode_images(self, jpg_paths, max_batch_size=1024):
        gpu_count = torch.cuda.device_count()

        jpg_paths_split = np.array_split(jpg_paths, gpu_count)
        manager = mp.Manager()
        outputs = manager.dict()
        processes = []
        ctx = get_context('spawn')
        print("starting processes for each gpu")
        for i in range(gpu_count):
            p = ctx.Process(target=self.encode_images_per_gpu, args=(list(jpg_paths_split[i]), i, max_batch_size, outputs))
            p.start()
            processes.append(p)
        print("started processes for each gpu")
        for p in processes:
            p.join()
        
        all_embeddings = []
        for i in range(gpu_count):
            embeddings = outputs[i]
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

class PDFsToEmbeddings:
    # def __init__(self, pdf_directory, txt_directory, jpgs_dir, e_jpgs_dir, embeddings_dir, embeddings_img_dir, embeddings_extract_dir, embedding_model):
    def __init__(self, pdf_directory, txt_directory, jpgs_dir, e_jpgs_dir, embeddings_dir, embeddings_img_dir, embeddings_extract_dir, metadata_dir, text_model, model_pool):
        self.pdfs_path = pdf_directory
        self.txts_path = txt_directory
        self.jpgs_path = jpgs_dir
        self.extracted_jpgs_path = e_jpgs_dir
        self.embeddings_path = embeddings_dir
        self.embeddings_img_path = embeddings_img_dir
        self.embeddings_img_e_path = embeddings_extract_dir
        self.metadata_dir = metadata_dir
        self.text_model = text_model  # TextEmbeddingModel
        self.model_pool = model_pool  # for multi-gpu use, this is the model_pool


    # *******************************************************************************************************************
    # this is the og dif pdf -> dir txt -> dir embed pipeline 
    # *******************************************************************************************************************

    # (1) pdf -> txt 

    # converts a single pdf file to a txt files (one txt per page)
    @staticmethod
    def convert_pdf_to_txt_and_img(txts_path, imgs_path, pdfs_path, pdf_file):
        # print("Paginating & Scraping Text: " + pdf_file)
        pdf_path = os.path.join(pdfs_path, pdf_file)
        #subdir for each pdf 
        pdf_txt_subdir = os.path.join(txts_path, os.path.splitext(pdf_file)[0])
        pdf_img_subdir = os.path.join(imgs_path, os.path.splitext(pdf_file)[0])

        os.makedirs(pdf_txt_subdir, exist_ok=True)
        os.makedirs(pdf_img_subdir, exist_ok=True)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = []
                for page in pdf.pages:
                    text.append(page.extract_text())
                images = []
                for page in pdf.pages:
                    images.append(page.to_image(resolution=72))
        except:
            text = ["",]
            
        for page_num, page_text in enumerate(text):
            txt_file_path = os.path.join(pdf_txt_subdir, f'{os.path.splitext(pdf_file)[0]}_{page_num}.txt')
            if len(page_text) != 0:
                with open(txt_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(page_text)
            
            img_file_path = os.path.join(pdf_img_subdir, f'{os.path.splitext(pdf_file)[0]}_{page_num}.jpeg')
            image = images[page_num]
            with open(img_file_path, 'w', encoding='utf-8') as text_file:
                image.save(img_file_path, format="png")
            
    # converts dir of pdfs -> dir of subdirs of txt files of each page AKA OVERALL PDFS -> TXTS
    # 1. OG VERSION 
    # def convert_pdfs_to_txt(self, pdf_files):
    #     self.ensure_dir(self.txts_path)
    #     for pdf_file in pdf_files:
    #         self.convert_pdf_to_txt(pdf_file)

    # 2. MP VERSION (from kyle)
    def convert_pdfs_to_txt_and_img(self, pdf_files=None):
        self.ensure_dir(self.txts_path)
        if pdf_files is None:
            pdf_files = os.listdir(self.pdfs_path)
        ctx = get_context('spawn')
        with ctx.Pool(processes=os.cpu_count()) as pool:
            pool.starmap(self.convert_pdf_to_txt_and_img, [(self.txts_path, self.jpgs_path, self.pdfs_path, file) for file in pdf_files])

    # *******************************************************************************************************************
    # 1. this is the dir pdf -> dir img (of entire page) -> dir embed (of entire page) shared with og embed dir
    # *******************************************************************************************************************

    def natural_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    # pdfs -> dir of subdirs of extracted images
    def convert_pdfs_to_single_jpg(self, pdf_files):
        parser = PdfToJpeg(self.pdfs_path, self.jpgs_path, 100)
        parser.convert_directory_to_jpegs(pdf_files)
    
    @staticmethod
    def convert_img_embedding_to_files_batch(embed_and_paths):
        embed, embed_file_paths = embed_and_paths
        for output_path, embedding in zip(embed_file_paths, embed):
            file_name = output_path.replace('.jpg', '.npy')
            # print(f"img file_name: {file_name} has been saved.")
            np.save(file_name, embedding)
    
    def convert_img_embedding_to_files(self, embed, embed_file_paths):
        # split the embedding up into chunks
        chunks = np.array_split(embed, os.cpu_count())
        # chunks = np.array_split(embed, 2)
        chunk_embed_file_paths = []

        start = 0
        for i in range(len(chunks)):
            end = chunks[i].shape[0]
            chunk_embed_file_paths.append(embed_file_paths[start: start + end])
            start = start + end 
        
        if len(chunks) != len(chunk_embed_file_paths):
            raise Exception("chunks and chunk_embed_file_paths should be the same length.")

        ctx = get_context('spawn')
        with ctx.Pool(processes=os.cpu_count()) as pool:
        # with ctx.Pool(processes=2) as pool:
            pool.map(self.convert_img_embedding_to_files_batch, zip(chunks, chunk_embed_file_paths))
    
    # *******************************************************************************************************************
    # dir pdf --> dir img (extracted) -> dir embed (extracted) shared with og embed dir
    # *******************************************************************************************************************

    # single pdf -> extracted img, extracted img embedding (using og embed dir)  
    # multi gpu extract images below 
    @staticmethod
    def extract_img_pdfs(pdf_directory, extracted_jpgs_path, embeddings_img_e_path, pdf_path):
        full_pdf_path = Path(pdf_directory) / Path(pdf_path)
        output_img_dir_path = Path(extracted_jpgs_path) / Path(pdf_path).stem
        output_img_dir_path.mkdir(parents=True, exist_ok=True)

        try:
            with fitz.open(full_pdf_path) as pdf_doc:
                title = os.path.splitext(os.path.basename(pdf_path))[0]

                empty = True
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    for i, img in enumerate(page.get_images(full=True)):
                        if i == 4:  # four images max per page extracted
                            break
                        empty = False

                        xref = img[0]
                        image_dict = pdf_doc.extract_image(xref)
                        image_bytes = image_dict["image"]

                        try:
                            image = Image.open(io.BytesIO(image_bytes))
                            image.load()
                        except Exception as e:
                            continue

                        image_path = Path(output_img_dir_path) / f"{title}_{page_num}_{i}.jpg"
                        image = image.convert("RGB")

                        if image.size[0] < 80 or image.size[1] < 80 or image.size[0] > 7000 or image.size[1] > 7000:  #image is too small/big to be considered
                            continue
                        image.save(image_path, "JPEG")
                
                if empty:
                    shutil.rmtree(output_img_dir_path)

        except Exception as e:
            logging.error(f"can't open PDF {pdf_path}: {e}")
            return

    
    def convert_pdfs_to_extracted_imgs(self, pdf_files):
        ctx = get_context('spawn')
        with ctx.Pool(processes=os.cpu_count()) as pool:
            pool.starmap(self.extract_img_pdfs, [(self.pdfs_path, self.extracted_jpgs_path, self.embeddings_img_e_path, file) for file  in pdf_files])


    # *******************************************************************************************************************
    # pdf --> dir metadata (json) for each pdf
    # *******************************************************************************************************************
    def create_metadata_jsons(self, pdf_files):
        os.makedirs(self.metadata_dir, exist_ok=True)
        for pdf_file in pdf_files:
            json_data = dict()
            pdf_path = os.path.join(self.pdfs_path, pdf_file)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    num_pages = len(pdf.pages)
                    gov_name = pdf.metadata.get('Title', 'Unknown')
                    timestamp = pdf.metadata.get('CreationDate', 'Unknown')
                    json_data['gov_name'] = gov_name
                    json_data['timestamp'] = timestamp
                    json_data['num_pages'] = num_pages
            except Exception as e:
                json_data['gov_name'] = 'Unknown'
                json_data['timestamp'] = 'Unknown'
                json_data['num_pages'] = 1
                print(f"Skipping invalid PDF {pdf_path}: {e}")
                continue
            
            pdf_metadata_dir = os.path.join(self.metadata_dir, pdf_file)
            os.makedirs(pdf_metadata_dir, exist_ok=True)
            json_file_path = os.path.join(pdf_metadata_dir, "metadata.json")
            with open(json_file_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

    # *******************************************************************************************************************
    # overall pipeline
    # *******************************************************************************************************************

    # given list of pdf_files. TODO: update code to handle case where pdf_files is None
    def pdfs_to_embeddings(self, pdf_files=None):
        pdf_files = pdf_files or os.listdir(self.pdfs_path)
        time1 = time.time()

        print("Converting pdfs to txts and page images")
        self.convert_pdfs_to_txt_and_img(pdf_files)
        time2 = time.time()

        print("Converting txts to embeddings")
        compute_text_embeddings(self.text_model, self.model_pool, self.txts_path, self.embeddings_path)
        time3 = time.time()
        time4 = time.time()
        
        img_paths = []
        embedding_paths = []
        for img_subdir in os.scandir(self.jpgs_path):
            if img_subdir.is_dir():
                img_subdir_paths = os.listdir(img_subdir.path)
                for img_file in img_subdir_paths:
                    img_paths.append(os.path.join(img_subdir.path, img_file))
                    embedding_paths.append(os.path.join(self.embeddings_img_path, os.path.splitext(img_file)[0] + '.npy'))

        print("Embedding this many images: ", len(img_paths))
        img_model = CLIPEmbeddingModel()
        emb = img_model.encode_images(img_paths)

        print("Embeddings computed. Shape:", emb.shape)
        self.convert_img_embedding_to_files(emb, embedding_paths)
        time5 = time.time()

        print("Converting pdfs to extracted imgs and embds")
#        self.convert_pdfs_to_extracted_imgs(pdf_files)  # extract images and save
#        extract_img_paths, extract_all_embed_file_paths = self.convert_imgs_to_embeddings(self.embeddings_img_e_path, self.extracted_jpgs_path)
#        emb_e = img_model.encode_images(extract_img_paths)
#        self.convert_img_embedding_to_files(emb_e, extract_all_embed_file_paths)
        time6 = time.time()


        print("Converting pdfs to extracted imgs and embds")
        self.create_metadata_jsons(pdf_files)  # extract images and save
        time7 = time.time()

        first = time2 - time1
        sec = time3 - time2
        third = time4 - time3
        fourth = time5 - time4
        fifth = time6 - time5
        sixth = time7 - time6

        print("pdf -> txt time: ", first)
        print("txt -> embed time: ", sec)
        print("pdf -> img per page time: ", third)
        print("img per page -> embed time: ", fourth)
        print("extracted img -> embed time: ", fifth)
        print("pdf -> json time: ", sixth)

        return first, sec, third, fourth, fifth, sixth

    # *******************************************************************************************************************
    # helper functions
    # *******************************************************************************************************************

    # makes sure that the directory specified is created
    @staticmethod
    def ensure_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
