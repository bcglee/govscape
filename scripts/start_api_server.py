import govscape as gs
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start the GovScape API server')
    parser.add_argument('-p', '--pdf-directory', default='data/test_data/TechnicalReport234PDFs', help='Directory containing PDF files')
    parser.add_argument('-d', '--data-directory', default='data/test_data', help='Directory containing data files')
    parser.add_argument('-m', '--model', default='CLIP', choices=['CLIP', 'UAE'], help='The model to use for embedding.')
    parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of top results to return')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode')
    
    args = parser.parse_args()
    
    pdf_directory = args.pdf_directory
    txt_directory =  args.data_directory + '/txt'
    embeddings_directory =  args.data_directory + '/embeddings'
    index_directory =  args.data_directory + '/index'
    image_directory =  args.data_directory + '/images'
    
    if args.model == "CLIP":
        model = gs.CLIPEmbeddingModel()
    elif args.model == "UAE":
        model = gs.TextEmbeddingModel()

    index_config = gs.IndexConfig(pdf_directory, embeddings_directory, index_directory, image_directory)
    server_config = gs.ServerConfig(index_config, gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, image_directory, model), k=args.top_k)
    
    server = gs.Server(server_config)
    server.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
