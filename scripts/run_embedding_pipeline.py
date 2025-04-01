import govscape as gs
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
                        prog='EmbeddingPipeline',
                        description='Runs the govscape embedding pipeline. This performs OCR, embedding, index construction, and page-jpeg generation.',
                        epilog='')
    parser.add_argument('-p', '--pdf-directory', default='', help='The directory holding the pdfs that you\'d like to embed.')      # option that takes a value
    parser.add_argument('-d', '--data-directory', default='', help='The directory where the embeddings and other metadata should be stored.')      # option that takes a value
    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
    args = parser.parse_args()
    
    pdf_directory = args.pdf_directory
    txt_directory =  args.data_directory + '/txt'
    embeddings_directory =  args.data_directory + '/embeddings'
    index_directory =  args.data_directory + '/index'
    image_directory =  args.data_directory + '/images'

    processor = gs.PDFsToEmbeddings(pdf_directory, txt_directory, embeddings_directory, gs.CLIPEmbeddingModel())
    processor.pdfs_to_embeddings()

    bin_file = os.path.join(embeddings_directory, "embeddings.bin")
    npytobin = gs.NpyToBin(bin_file)
    for subdir in os.listdir(embeddings_directory):
        subdir_path = os.path.join(embeddings_directory, subdir)
        if os.path.isdir(subdir_path):
            npytobin.convert_pdfdir_to_bin(subdir_path)
    
    indexConfig = gs.IndexConfig(pdf_directory, embeddings_directory, index_directory, image_directory, "Disk")
    indexBuilder = gs.IndexBuilder(indexConfig)
    indexBuilder.build_index()

    pdftojpeg = gs.PdfToJpeg()
    pdftojpeg.convert(pdf_directory, image_directory)

if __name__ == '__main__':
         main()