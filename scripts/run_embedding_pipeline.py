import govscape as gs
import argparse
import os

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
    processor = gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, image_directory, model)
    processor.pdfs_to_embeddings()

<<<<<<< HEAD
    bin_file = os.path.join(embeddings_directory, "embeddings.bin")
    page_indices = os.path.join(embeddings_directory, "page_indices.bin")
    npytobin = gs.NpyToBin(bin_file, page_indices)
    for subdir in os.listdir(embeddings_directory):
        subdir_path = os.path.join(embeddings_directory, subdir)
        if os.path.isdir(subdir_path):
            npytobin.convert_pdfdir_to_bin(subdir_path)
    
    indexConfig = gs.IndexConfig(pdf_directory, embeddings_directory, index_directory, image_directory, "Disk")
    indexBuilder = gs.IndexBuilder(indexConfig)
    indexBuilder.build_index()

    pdftojpeg = gs.PdfToJpeg()
    pdftojpeg.convert(pdf_directory, image_directory)
=======
    pdftojpeg = gs.PdfToJpeg(pdf_directory, image_directory, 100)
    pdftojpeg.convert_directory_to_jpegs()
>>>>>>> 3d20c10835e217ccd36aca7f6dd9a87b22dea546

if __name__ == '__main__':
         main()