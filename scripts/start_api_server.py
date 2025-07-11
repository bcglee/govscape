import govscape as gs
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start the GovScape API server')
    parser.add_argument('-p', '--pdf-directory', default='data/test_data/TechnicalReport234PDFs', help='Directory containing PDF files')
    parser.add_argument('-d', '--data-directory', default='data/test_data', help='Directory containing data files')
    parser.add_argument('-m', '--model', default='UAE', choices=['CLIP', 'UAE'], help='The model to use for embedding.')
    parser.add_argument('-k', '--top-k', type=int, default=20, help='Number of top results to return')
    parser.add_argument('-i', '--index_type', default='Memory', help='The type of index of use')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode')
    
    args = parser.parse_args()
    
    pdf_directory = args.pdf_directory
    txt_directory, embeddings_directory, index_directory, images_directory, metadata_directory = [
        getattr(args, f"{dir_name}_directory") or f"{args.data_directory}/{dir_name}"
        for dir_name in ['txt', 'embeddings', 'index', 'images', 'metadata']
    ]
    
    if args.model == "CLIP":
        model = gs.CLIPEmbeddingModel()
    elif args.model == "UAE":
        model = gs.TextEmbeddingModel()

    index_config = gs.IndexConfig(pdf_directory, embeddings_directory, index_directory, images_directory, metadata_directory, args.index_type)
    
    if args.index_type == 'Disk':
        i.load_index()
    
    server_config = gs.ServerConfig(
        index_config, 
        gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, images_directory, model), 
        k=args.top_k
    )
    server = gs.Server(server_config)
    server.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
