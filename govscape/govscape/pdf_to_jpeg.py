# The purpose of this file is to create a class that converts
# pdf pages to be compressed jpegs

from pdf2image import convert_from_path
import os
from multiprocessing import Pool

# wrapper class
class PdfToJpeg:
    def __init__(self, pdf_directory, save_directory, dpi):
        self.pdf_directory = pdf_directory
        self.save_directory = save_directory
        self.dpi = dpi
        pass

    def convert_pdf_to_jpeg(self, pdf_filename):

        print("Creating JPEG (DPI " + str(self.dpi) + "):" + pdf_filename)

        # creates a new directory for this pdf in save_directory
        pdf_basename = os.path.splitext(os.path.basename(pdf_filename))[0]
        pdf_directory = os.path.join(self.save_directory, pdf_basename)
        os.makedirs(pdf_directory, exist_ok=True)

        # saves each page into created directory
        pages = convert_from_path(pdf_filename, dpi=self.dpi)
        for i, page in enumerate(pages):
            output_path = os.path.join(self.save_directory, f"{pdf_basename}/{pdf_basename}_{i}.jpg")
            page.save(output_path, "JPEG")
        return None


    # pdf_directory -> directory to source pdfs
    # save_directory -> directory to save images
    def convert_directory_to_jpegs(self):
        # recursively finds the pdfs in pdf_directory
        pdf_files = []
        for root, _, files in os.walk(self.pdf_directory):
            for filename in files:
                pdf_files.append(os.path.join(root, filename))

        with Pool(processes=12) as pool:
            pool.map(self.convert_pdf_to_jpeg, pdf_files)




