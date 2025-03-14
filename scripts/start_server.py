import govscape as gs
import argparse

def main():
    parser = argparse.ArgumentParser(
                        prog='EmbeddingPipeline',
                        description='Runs the govscape embedding pipeline. This performs OCR, embedding, index construction, and page-jpeg generation.',
                        epilog='')
    parser.add_argument('-p', '--pdf-directory', default='data/test_data/TechnicalReport234PDFs', help='The directory holding the pdfs that you\'d like to embed.')      # option that takes a value
    parser.add_argument('-d', '--data-directory', default='data/test_data', help='The directory where the embeddings and other metadata should be stored.')      # option that takes a value
    parser.add_argument('-m', '--model', default='CLIP', help='The model to use for embedding.')      # option that takes a value
    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
    args = parser.parse_args()
    pdf_directory = args.pdf_directory
    txt_directory =  args.data_directory + '/txt'
    embeddings_directory =  args.data_directory + '/embeddings'
    index_directory =  args.data_directory + '/index'
    image_directory =  args.data_directory + '/images'
    model = None
    if args.model == "CLIP":
            model = gs.CLIPEmbeddingModel()
    elif args.model == "UAE":
            model = gs.TextEmbeddingModel()
    index_config = gs.IndexConfig(pdf_directory, embeddings_directory, index_directory, image_directory)
    server_config = gs.ServerConfig(index_config, gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, image_directory, model), k = 5)
    s = gs.Server(server_config)
    s.serve()

if __name__ == '__main__':
         main()