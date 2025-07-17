import govscape as gs
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start the GovScape API server')
    parser.add_argument('-p', '--pdf-directory', default='data/test_data/TechnicalReport234PDFs', help='Directory containing PDF files')
    parser.add_argument('-d', '--data-directory', default='data/test_data', help='Directory containing data files')
    parser.add_argument('-tm', '--text_model', default='UAE', help='The model to use for text embedding.')
    parser.add_argument('-vm', '--visual_model', default='CLIP', help='The model to use for visual embedding.')
    parser.add_argument('-k', '--top-k', type=int, default=20, help='Number of top results to return')
    parser.add_argument('-i', '--index_type', default='Memory', help='The type of index of use')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode')
    
    args = parser.parse_args()
    
    pdf_directory = args.pdf_directory
    if args.text_model == 'SentenceTransformer':
        text_model = gs.TextEmbeddingModel()
    else:
        raise ValueError(f"Unsupported text model: {args.text_model}")
    if args.visual_model == 'CLIP':
        visual_model = gs.CLIPEmbeddingModel()
    else:
        raise ValueError(f"Unsupported visual model: {args.visual_model}")

    index_config = gs.IndexConfig(pdf_directory, args.data_directory, args.index_type)
    
    if args.index_type == 'Disk':
        i.load_index()
    
    server_config = gs.ServerConfig(
        index_config, 
        text_model,
        visual_model,
        k=args.top_k
    )
    
    server = gs.Server(server_config)
    server.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
