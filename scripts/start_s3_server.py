import govscape as gs
import argparse
from govscape.server_s3 import S3Server

def main():
    parser = argparse.ArgumentParser(
                    prog='S3SearchServer',
                    description='Runs the govscape search server using pre-processed data from S3.',
                    epilog='')
    parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Verify S3 embedding files before starting server')
    args = parser.parse_args()
    
    index_config = gs.IndexConfig()
    server_config = gs.ServerConfig(index_config, gs.CLIPEmbeddingModel(), k=args.top_k)
    
    s = S3Server(server_config)
    if args.verify:
        s3_loader = gs.S3DataLoader()
        s3_loader.verify_embeddings(n_samples=10)
        return
    s.serve()

if __name__ == '__main__':
    main() 