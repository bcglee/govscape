# pip install torch
# pip install transformers 
# pip install numpy

# for pdf -> jpg
from PIL import Image
from io import BytesIO
import os
import fitz
# for CLIP
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
#for saving embeddings
import numpy as np

# 1. extract images of PDF files -> and outputs them to .jpg
#   - has stucture: dir -> subdir for each PDF -> .jpg files of each page 
# 2. Plug in .jpg files to CLIP to generate embeddings -> save as .npy file (could implement .pt if needed)
#   - has structure: dir -> subdir for each PDF -> .npy files for each page embedding 

def main():
    processor = JPGsToEmbeddings("test_data/pdfs", "test_data/jpgs", "test_data/embedding")
    processor.pdfs_to_embeddings()

class JPGsToEmbeddings:
    def __init__(self, pdf_directory, jpg_directory, embeddings_dir):
        self.jpgs_path = jpg_directory
        self.pdfs_path = pdf_directory
        self.embeddings_path = embeddings_dir
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #1. PDF -> JPG

    # converts a dir of pdfs to a dir of subdirs for each pdf of jpg files of each page 
    def convert_pdfs_to_jpg(self):
        if not os.path.exists(self.jpgs_path):
            os.makedirs(self.jpgs_path)
        
        pdf_files = os.listdir(self.pdfs_path)

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdfs_path, pdf_file)
            
            #subdir for each pdf 
            pdf_subdir = os.path.join(self.jpgs_path, os.path.splitext(pdf_file)[0])
            if not os.path.exists(pdf_subdir):
                os.makedirs(pdf_subdir)
            
            pdf = fitz.open(pdf_path)

            for page_num, page in enumerate(pdf):
                images = page.get_images(full=True)
                for img_ind, img in enumerate(images):
                        # way to reference the image
                        img_ref = img[0]
                        base_img = pdf.extract_image(img_ref)
                        img_bytes = base_img["image"]

                        new_jpg = Image.open(BytesIO(img_bytes))

                        img_file_path = os.path.join(pdf_subdir, f'{os.path.splitext(pdf_file)[0]}_{page_num}_{img_ind}.jpg')
                        new_jpg.save(img_file_path, "JPEG")


    # takes in a single jpg file and converts it into an embedding
    def convert_jpg_to_embedding(self, jpg_path):
        image = Image.open(jpg_path)

        # preprocess image
        inputs = self.processor(images = image, return_tensors="pt")

        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        return image_embedding

    # converts a dir of subdirs for each image for each page into a dir of subdir of embeddings in .npy
    def convert_imgs_to_embeddings(self):
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)

        jpg_subdirs_paths = []
        for jpg_subdir in os.scandir(self.jpgs_path):
            if jpg_subdir.is_dir():
                jpg_subdirs_paths.append(jpg_subdir.path)
        
        for jpg_subdir_path in jpg_subdirs_paths:

            #making the subdir that will hold the embeddings for each PDF 
            embed_name = os.path.basename(jpg_subdir_path)
            embedding_dir = os.path.join(self.embeddings_path, embed_name)

            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)

            #all jpg files in the jpg subdir input 
            jpg_files = os.listdir(jpg_subdir_path)

            for jpg_file in jpg_files:
                jpg_path = os.path.join(jpg_subdir_path, jpg_file)
                
                #check if file is empty; just skip to next if true 
                if os.stat(jpg_path).st_size == 0:
                    continue

                embedding = self.convert_jpg_to_embedding(jpg_path)

                file_name = os.path.splitext(jpg_file)[0] + ".npy"
                output_path = os.path.join(embedding_dir, file_name)
                np.save(output_path, embedding.cpu().numpy())
    
    # 1 + 2
    #converts a dir of pdfs to a dir of embeddings of .npy
    def pdfs_to_embeddings(self):
        self.convert_pdfs_to_jpg()
        self.convert_imgs_to_embeddings()

#test:

#please write your file paths - don't forget \\ instead of \
'''pdf_directory = ""
txt_directory = ""
embeddings_directory = ""

processor = PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory)
processor.pdfs_to_embeddings()'''


#test:
'''
test = PDFsToEmbeddings("", "", "")
print(test.text_to_embeddings("hello"))
'''

if __name__=="__main__":
    main()