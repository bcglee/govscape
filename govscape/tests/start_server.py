import govscape as gs
import argparse
import pytest
import os
from argparse import Namespace

@pytest.mark.usefixtures()
def test_args():
    return Namespace(
        pdf_directory='data/test_data/TechnicalReport234PDFs',
        data_directory='data/test_data',
        model='UAE',
        verbose=False,
        index = "Memory"
    )

def serve(file, server):
    with open(file, 'r', encoding='utf-8') as f:
            pdf = f.readline()
            correct = 0
            while pdf:
                for _ in range(0, 10):
                    search = f.readline()
                    print(search)

                    results = server.search(search)

                    for result in results:
                        if pdf in result["pdf"]:
                            correct += 1
                print(pdf + " " + str(correct))
                pdf = f.readline()
                correct = 0

def test_server(test_args):
    pdf_directory = test_args.pdf_directory
    txt_directory = os.path.join(test_args.data_directory, 'txt')
    embeddings_directory = os.path.join(test_args.data_directory, 'embeddings')
    index_directory = os.path.join(test_args.data_directory, 'index')
    image_directory = os.path.join(test_args.data_directory, 'images')
    index_type = test_args.index

    if test_args.model == "CLIP":
        model = gs.CLIPEmbeddingModel()
    elif test_args.model == "UAE":
        model = gs.TextEmbeddingModel()
    else:
        raise ValueError("Unsupported model type")
    index_config = gs.IndexConfig(pdf_directory, embeddings_directory, index_directory, image_directory, test_args.index)
    server_config = gs.ServerConfig(index_config, gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, image_directory, model), k = 5)
    s = gs.Server(server_config)
    serve('data/test_data/queries/test.txt', s)

if __name__ == '__main__':
         main()