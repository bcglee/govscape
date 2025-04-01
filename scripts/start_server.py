import govscape as gs
import argparse

def main():
    parser = argparse.ArgumentParser(
                        prog='EmbeddingPipeline',
                        description='Runs the govscape embedding pipeline. This performs OCR, embedding, index construction, and page-jpeg generation.',
                        epilog='')
    parser.add_argument('-p', '--pdf-directory', default='', help='The directory holding the pdfs that you\'d like to embed.')      # option that takes a value
    parser.add_argument('-d', '--data-directory', default='', help='The directory where the embeddings and other metadata should be stored.')      # option that takes a value
    parser.add_argument('-i', '--index_type', default='Memory', help='The type of index of use')
    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
    args = parser.parse_args()
    pdf_directory = args.pdf_directory
    txt_directory =  args.data_directory + '/txt'
    embeddings_directory =  args.data_directory + '/embeddings'
    index_directory =  args.data_directory + '/index'
    image_directory =  args.data_directory + '/images'
    index_config = gs.IndexConfig(pdf_directory, embeddings_directory, index_directory, image_directory, args.index_type)
    i = gs.IndexBuilder(index_config)
    i.load_index()
    server_config = gs.ServerConfig(index_config, gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, gs.CLIPEmbeddingModel()), i)
    s = gs.Server(server_config)
    s.serve()

if __name__ == '__main__':
         main()