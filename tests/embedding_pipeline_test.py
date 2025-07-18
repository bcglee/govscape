# pip install pytest
# pip install reportlab 

import pytest 
import sys  # to find pdf_to_embedding.py
sys.path.append('/homes/gws/cgong16/govscape/govscape/govscape/') # insert path to the pdf_to_embedding.py file here
from reportlab.pdfgen import canvas
from pdf_to_embedding import * 
from reportlab.lib.pagesizes import letter
import os

model = CLIPEmbeddingModel()
embed_pipeline = PDFsToEmbeddings('test_files/pdfs', 'test_files/text', 'test_files/embeddings', model)
pdf_directory = 'test_files/pdfs'
txt_directory = 'test_files/text'
embed_directory = 'test_files/embeddings'

# checks that the text outputs an embedding of type np.ndarray
def test_encode_text_CLIP():
    text = "test"
    embedding = model.encode_text(text)
    assert isinstance(embedding, np.ndarray)

# checks that the pdf text matches the text outputted
def test_convert_pdf_to_text():
    text = embed_pipeline.convert_pdf_to_text('test_files/pdfs/govscape_intro.pdf')
    assert text == ["hello my name is govscape"]

# makes sure the number of subdirs are equal to each other, 
# also that there are more than one txt file per subdir
def test_convert_pdfs_to_txt():
    embed_pipeline.convert_pdfs_to_txt()
    assert os.path.isdir(txt_directory) == True

    pdfs = [pdf for pdf in os.listdir(pdf_directory) if pdf.endswith(".pdf")]

    subdirs_txt = [os.path.join(txt_directory, d) for d in os.listdir(txt_directory) if os.path.isdir(os.path.join(txt_directory, d))]
        
    assert(len(pdfs) == len(subdirs_txt))

    for subdir in subdirs_txt:
        txt_files = [f for f in os.listdir(subdir) if f.endswith(".txt")]
        assert len(txt_files) > 0

# checks that the text creates an embedding of type np.ndarray
def test_text_to_embeddings():
    text = "test"
    embedding = embed_pipeline.text_to_embeddings(text)
    assert isinstance(embedding, np.ndarray)

# checks that the txt file creates an embedding of type np.ndarray
def test_convert_txt_to_embedding():
    txt_file_paths = []

    for subdir, _, files in os.walk(txt_directory):
        for file in files:
            if file.endswith(".txt"):
                txt_file_paths.append(os.path.join(subdir, file))
    
    for txt_path in txt_file_paths:
        embedding = embed_pipeline.convert_txt_to_embedding(txt_path)
        assert isinstance(embedding, np.ndarray)

# checks for the same file structure with txt and embed and that they have the same number of embeddings and txt files
def test_convert_txts_to_embeddings():
    embed_pipeline.convert_txts_to_embeddings()

    subdirs_txt = [os.path.join(txt_directory, d) for d in os.listdir(txt_directory) if os.path.isdir(os.path.join(txt_directory, d))]
    subdirs_npy = [os.path.join(embed_directory, d) for d in os.listdir(embed_directory) if os.path.isdir(os.path.join(embed_directory, d))]
    assert(len(subdirs_txt) == len(subdirs_npy))
    
    npy_counts = {}
    for subdir in subdirs_npy:
        npy_count = sum(1 for filename in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, filename)))
        npy_counts[os.path.basename(subdir)] = npy_count

    for subdir in subdirs_txt:
        file_count = sum(1 for filename in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, filename)))
        assert((npy_counts[os.path.basename(subdir)] - 1) == file_count)  # -1 to account for the json file
    
# checks if same file structure and that the final embeddings contains at least two files (json and embedding file)
def test_pdfs_to_embeddings():
    embed_pipeline.pdfs_to_embeddings()
    subdirs_txt = [os.path.join(txt_directory, d) for d in os.listdir(txt_directory) if os.path.isdir(os.path.join(pdf_directory, d))]
    subdirs_npy = [os.path.join(embed_directory, d) for d in os.listdir(embed_directory) if os.path.isdir(os.path.join(embed_directory, d))]
    
    for subdir in subdirs_npy:
        npy_count = sum(1 for filename in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, filename)))
        assert(npy_count >= 2)


#TODO: add functionality that deletes the embedding and txt directories when finished