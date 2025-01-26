# pip install pdfplumber 
# pip install torch
# pip install transformers 
# pip install numpy

# for pdf -> txt
import pdfplumber 
import os
# for CLIP
from transformers import CLIPProcessor, CLIPModel
import torch
#for saving embeddings
import numpy as np

# 1. extract text of PDF files -> and outputs them to .txt 
#   - has stucture: dir -> subdir for each PDF -> .txt files of each page 
# 2. Plug in .txt files to CLIP to generate embeddings -> save as .npy file (could implement .pt if needed)
#   - has structure: dir -> subdir for each PDF -> .npy files for each page embedding 

class PDFsToEmbeddings:
    def __init__(self, pdf_directory, txt_directory, embeddings_dir):
        self.pdfs_path = pdf_directory
        self.txts_path = txt_directory
        self.embeddings_path = embeddings_dir
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #1. PDF -> TXT 

    # converts a single pdf file to text
    def _convert_pdf_to_text(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            text = []
            for page in pdf.pages:
                text.append(page.extract_text())
        return text

    # converts a dir of pdfs to a dir of subdirs for each pdf of txt files of each page 
    def _convert_pdfs_to_txt(self):
        if not os.path.exists(self.txts_path):
            os.makedirs(self.txts_path)
        
        pdf_files = os.listdir(self.pdfs_path)

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdfs_path, pdf_file)
            
            #subdir for each pdf 
            pdf_subdir = os.path.join(self.txts_path, os.path.splitext(pdf_file)[0])
            if not os.path.exists(pdf_subdir):
                os.makedirs(pdf_subdir)

            text = self._convert_pdf_to_text(pdf_path)

            for page_num, page_text in enumerate(text):
                txt_file_path = os.path.join(pdf_subdir, f'{os.path.splitext(pdf_file)[0]}_{page_num}.txt')
                with open(txt_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(page_text)

    #2. TXT -> CLIP EMBEDDINGS

    # takes in a single txt file and converts it into an embedding
    # added in some batch processing to make it faster
    def _convert_txt_to_embedding(self,txt_path):
        # need to convert .txt to text
        with open(txt_path, 'r') as file:
            text = file.read()
        
        #tokenize text
        inputs = self.processor(text=text, return_tensors="pt", padding=False, truncation=False)
        tokenized_text = inputs['input_ids'][0]
        
        #CLIP token limit = 77 so we have to divide into chunks and get embeddings for each of those
        max_chunk_len = 77
        text_chunks = []

        for i in range(0,len(tokenized_text), max_chunk_len):
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
        final_embedding = torch.mean(torch.stack(embeddings), dim=0)

        return final_embedding

    # converts a dir of subdirs for each pdf of txts for each page into a dir of subdir of embeddings in .npy
    def _convert_txts_to_embeddings(self):
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)

        txt_subdirs_paths = []
        for txt_subdir in os.scandir(self.txts_path):
            if txt_subdir.is_dir():
                txt_subdirs_paths.append(txt_subdir.path)
        
        for txt_subdir_path in txt_subdirs_paths:

            #making the subdir that will hold the embeddings for each PDF 
            embed_name = os.path.basename(txt_subdir_path)
            embedding_dir = os.path.join(self.embeddings_path, embed_name)

            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)

            #all txt files in the txt subdir input 
            txt_files = os.listdir(txt_subdir_path)

            for txt_file in txt_files:
                txt_path = os.path.join(txt_subdir_path, txt_file)
                
                #check if file is empty; just skip to next if true 
                if os.stat(txt_path).st_size == 0:
                    continue

                embedding = self._convert_txt_to_embedding(txt_path)

                file_name = os.path.splitext(txt_file)[0] + ".npy"
                output_path = os.path.join(embedding_dir, file_name)
                np.save(output_path, embedding.cpu().numpy())

    
    
    #takes in a string and converts it into an embedding. 
    def text_to_embeddings(self, text):
        #tokenize text
        inputs = self.processor(text=text, return_tensors="pt", padding=False, truncation=False)
        tokenized_text = inputs['input_ids'][0]
        
        #CLIP token limit = 77 so we have to divide into chunks and get embeddings for each of those
        max_chunk_len = 77
        text_chunks = []

        for i in range(0,len(tokenized_text), max_chunk_len):
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
        final_embedding = torch.mean(torch.stack(embeddings), dim=0)

        return final_embedding
    
    # 1 + 2
    
    def pdfs_to_embeddings(self):
        self._convert_pdfs_to_txt()
        self._convert_txts_to_embeddings()

#test:
'''
#please write your file paths 
pdf_directory = ""
txt_directory = ""
embeddings_directory = ""

processor = PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory)
processor.pdfs_to_embeddings()
'''

#test:
'''
test = PDFsToEmbeddings("", "", "")
print(test.text_to_embeddings("hello"))
'''