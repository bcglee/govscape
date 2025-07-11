import govscape as gs
import argparse
import boto3

def main():
    parser = argparse.ArgumentParser(description='Start the GovScape API server')
    parser.add_argument('-p', '--pdf-directory', default='data/test_data/TechnicalReport234PDFs', help='Directory containing PDF files')
    parser.add_argument('-d', '--data-directory', default='data/test_data', help='Directory containing data files')
    parser.add_argument('--txt-directory', default=None, help='Custom txt directory path')
    parser.add_argument('--embeddings-directory', default=None, help='Custom embeddings directory path')
    parser.add_argument('--index-directory', default=None, help='Custom index directory path')
    parser.add_argument('--images-directory', default=None, help='Custom images directory path')
    parser.add_argument('-m', '--model', default='CLIP', choices=['CLIP', 'UAE'], help='The model to use for embedding.')
    parser.add_argument('-k', '--top-k', type=int, default=20, help='Number of top results to return')
    parser.add_argument('-i', '--index_type', default='Memory', help='The type of index of use')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode')
    parser.add_argument('--s3-source-bucket', default="bcgl-public-bucket", help='Where to retrieve the data from in S3')
    parser.add_argument('--s3-source-folder', default="2008_EOT_PDFs/data_test_100k_final/", help='Where to retrieve the data from in S3')
    
    args = parser.parse_args()
    
    s3 = boto3.resource('s3')
    source_bucket = args.s3_source_bucket
    source_prefix = args.s3_source_folder
    data_directory = args.data_directory

    bucket = s3.Bucket(source_bucket)
    for obj in bucket.objects.filter(Prefix=source_prefix):
        src_key = obj.key
        dest_key = data_directory + '/' + src_key[len(source_prefix):]
        bucket.download_file(src_key, dest_key)
        

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
