import govscape as gs
import argparse
from govscape.pdf_to_embedding import CLIPEmbeddingModel

def main():
    parser = argparse.ArgumentParser(
                        prog='GovscapeServer',
                        description='Runs the govscape server for searching PDF embeddings.',
                        epilog='')
    
    # Add a mode argument to select between original and S3-downloaded modes
    parser.add_argument('--mode', choices=['original', 's3-downloaded'], default='original',
                        help='Server mode: original (local PDFs) or s3-downloaded (pre-downloaded embeddings)')
    
    # Original mode arguments
    parser.add_argument('-p', '--pdf-directory', default='', 
                        help='The directory holding the PDFs that you\'d like to embed (original mode).')
    parser.add_argument('-d', '--data-directory', default='', 
                        help='The directory where the embeddings and other metadata should be stored (original mode).')
    
    # S3-downloaded mode arguments
    parser.add_argument('-e', '--embeddings-dir', 
                        help='Directory containing downloaded embeddings (s3-downloaded mode).')
    parser.add_argument('-i', '--image-dir', default='', 
                        help='Directory containing JPEG images (s3-downloaded mode).')
    
    # Common arguments
    parser.add_argument('-k', '--top-k', type=int, default=5, 
                        help='Number of results to return.')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.mode == 'original':
        # Original mode - use local PDFs and generate embeddings
        if not args.pdf_directory or not args.data_directory:
            parser.error("--pdf-directory and --data-directory are required for original mode")
            
        pdf_directory = args.pdf_directory
        txt_directory = args.data_directory + '/txt'
        embeddings_directory = args.data_directory + '/embeddings'
        index_directory = args.data_directory + '/index'
        image_directory = args.data_directory + '/images'
        
        # Create model and configs
        clip_model = CLIPEmbeddingModel()
        pdf_to_embeddings = gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, clip_model)
        index_config = gs.IndexConfig(pdf_directory, embeddings_directory, index_directory, image_directory)
        server_config = gs.ServerConfig(index_config, pdf_to_embeddings, k=args.top_k)
        
        # Start original server
        s = gs.Server(server_config)
        s.serve()
        
    elif args.mode == 's3-downloaded':
        # S3-downloaded mode - use pre-downloaded embeddings
        if not args.embeddings_dir:
            parser.error("--embeddings-dir is required for s3-downloaded mode")
            
        # Create model and configs
        clip_model = CLIPEmbeddingModel()
        image_directory = args.image_dir
        
        # Create dummy config with image directory
        index_config = gs.IndexConfig("", "", "", image_directory)
        server_config = gs.ServerConfig(index_config, clip_model, k=args.top_k)
        
        # Start precomputed server with downloaded embeddings
        s = gs.PrecomputedServer(server_config, args.embeddings_dir)
        s.serve()

if __name__ == '__main__':
    main()