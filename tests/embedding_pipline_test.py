# pip install pytest
# pip install reportlab 

import pytest 
import sys  # to find pdf_to_embedding.py
sys.path.append('/homes/gws/cgong16/govscape/src/govscape') # insert path to the pdf_to_embedding.py file here
from reportlab.pdfgen import canvas
from pdf_to_embedding import * 
from reportlab.lib.pagesizes import letter
import os


'''TEST_FILES_PATH = '/test_files/pdfs/'

@pytest.fixture
def shared_data():
    #os.makedirs(os.path.dirname(TEST_FILES_PATH + "test1/"), exist_ok=True)
    #os.makedirs(os.path.dirname(TEST_FILES_PATH + "test2/"), exist_ok=True)

    #pdf_path1 = os.path.join(TEST_FILES_PATH, "test1", "test1.pdf")
    #pdf_path2 = os.path.join(TEST_FILES_PATH, "test2", "test2.pdf")

    create_pdf(pdf_path1, "hello my name is govscape.")
    create_pdf(pdf_path2, "what would you like to search for?")

    model = CLIPEmbeddingModel()
    embed_pipeline = PDFsToEmbeddings('test_files/pdfs', 'test_files/text', 'test_files/embeddings', model)

    data = {
        'pdf_path1' : pdf_path1, 
        'pdf_path2' : pdf_path2,
        'CLIP_model' : model,
        'embed_pipeline' : embed_pipeline
    }
    return data'''

model = CLIPEmbeddingModel()
embed_pipeline = PDFsToEmbeddings('test_files/pdfs', 'test_files/text', 'test_files/embeddings', model)
pdf_directory = 'test_files/pdfs'
txt_directory = 'test_files/text'
embed_directory = 'test_files/embeddings'
     
def test_encode_text_CLIP():
    text = "test"
    embedding = model.encode_text(text)
    assert isinstance(embedding, np.ndarray)
 
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

def test_text_to_embeddings():
    text = "test"
    embedding = embed_pipeline.text_to_embeddings(text)
    assert isinstance(embedding, np.ndarray)

def test_convert_txt_to_embedding():
    txt_file_paths = []

    for subdir, _, files in os.walk(txt_directory):
        for file in files:
            if file.endswith(".txt"):
                txt_file_paths.append(os.path.join(subdir, file))
    
    for txt_path in txt_file_paths:
        embedding = embed_pipeline.convert_txt_to_embedding(txt_path)
        assert isinstance(embedding, np.ndarray)

def test_convert_txts_to_embeddings():
    embed_pipeline.convert_txts_to_embeddings()

    subdirs_txt = [os.path.join(txt_directory, d) for d in os.listdir(txt_directory) if os.path.isdir(os.path.join(txt_directory, d))]
    subdirs_npy = [os.path.join(embed_directory, d) for d in os.listdir(embed_directory) if os.path.isdir(os.path.join(embed_directory, d))]
    assert(len(subdirs_txt) == len(subdirs_npy))

    for subdir in subdirs_npy:
        print(subdir)
        npy_files = [f for f in os.listdir(subdir) if f.endswith(".npy")]

        if len(npy_files) == 0:
            txt_subdir_path = os.path.join(os.path.dirname(os.path.dirname(subdir)), 'text', os.path.basename(subdir).split('/', 1)[-1])

            for file_name in os.listdir(txt_subdir_path):
                if file_name.endswith('.txt'):
                    txt_file_path = os.path.join(txt_subdir_path, file_name)
                    assert(os.stat(txt_file_path).st_size == 0)
        else:
            assert len(npy_files) > 0


def test_pdfs_to_embeddings():
    return NotImplemented
    

