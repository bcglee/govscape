# The purpose of this file is to create a class that converts
# pdf pages to be compressed jpegs

from pdf2image import convert_from_path
import os

# wrapper class
class PdfToJpeg:
    def __init__(self):
        pass

    # pdf_directory -> directory to source pdfs
    # save_directory -> directory to save images
    def convert(self, pdf_directory, save_directory, dpi):

        # recursively finds the pdfs in pdf_directory
        pdfs = []
        for root, _, files in os.walk(pdf_directory):
            for filename in files:
                pdfs.append(os.path.join(root, filename))

        for pdf in pdfs:
            # converts each pdf into a page with 50 dots per inch
            pages = convert_from_path(pdf, dpi=dpi)

            # creates a new directory for this pdf in save_directory
            pdf_basename = os.path.splitext(os.path.basename(pdf))[0]
            pdf_directory = os.path.join(save_directory, pdf_basename)
            os.makedirs(pdf_directory, exist_ok=True)

            # saves each page into created directory
            for i, page in enumerate(pages):
                output_path = os.path.join(save_directory, f"{pdf_basename}/{pdf_basename}_{i}.jpg")
                page.save(output_path, "JPEG")




