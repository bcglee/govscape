import govscape as gs
import argparse

def main():
    parser = argparse.ArgumentParser(
                        prog='EmbeddingPipeline',
                        description='Runs the govscape embedding pipeline. This performs OCR, embedding, index construction, and page-jpeg generation.',
                        epilog='')
    parser.add_argument('-p', '--pdf-directory', default='data/test_data/TechnicalReport234PDFs', help='The directory holding the pdfs that you\'d like to embed.')      # option that takes a value
    parser.add_argument('-d', '--data-directory', default='data/test_data', help='The directory where the embeddings and other metadata should be stored.')      # option that takes a value
    parser.add_argument('-tm', '--text_model', default='UAE', help='The model to use for text embedding.')
    parser.add_argument('-vm', '--visual_model', default='CLIP', help='The model to use for visual embedding.')
    parser.add_argument('-i', '--index_type', default='Memory', help='The type of index of use')
    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
    args = parser.parse_args()
    pdf_directory = args.pdf_directory
    txt_directory, embeddings_directory, index_directory, images_directory, metadata_directory = [
        getattr(args, f"{dir_name}_directory") or f"{args.data_directory}/{dir_name}"
        for dir_name in ['txt', 'embeddings', 'index', 'images', 'metadata']
    ]
    
    if args.text_model == 'SentenceTransformer':
        text_model = gs.TextEmbeddingModel()
    else:
        raise ValueError(f"Unsupported text model: {args.text_model}")
    if args.visual_model == 'CLIP':
        visual_model = gs.CLIPEmbeddingModel()
    else:
        raise ValueError(f"Unsupported visual model: {args.visual_model}")

    index_config = gs.IndexConfig(pdf_directory, embeddings_directory, index_directory, images_directory, metadata_directory, args.index_type)
    server_config = gs.ServerConfig(index_config, gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, images_directory, metadata_directory, text_model, visual_model))
    s = gs.Server(server_config)
    s.serve()

if __name__ == '__main__':
         main()