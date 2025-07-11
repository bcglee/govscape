# The purpose of this file is to create a class that converts
# pdf pages to be compressed jpegs

from pdf2image import convert_from_path
import os
import shutil
from torch.multiprocessing import Pool, TimeoutError, get_context

# wrapper class
class PdfToJpeg:
    def __init__(self, pdf_directory, save_directory, dpi):
        self.pdf_directory = pdf_directory
        shutil.rmtree(save_directory, ignore_errors=True)
        os.makedirs(save_directory, exist_ok=True)
        self.save_directory = save_directory
        self.dpi = dpi
        pass

    def convert_pdf_to_jpeg(self, pdf_filename):

        print("Creating JPEG (DPI " + str(self.dpi) + "):" + pdf_filename)

        # creates a new directory for this pdf in save_directory
        pdf_basename = os.path.splitext(os.path.basename(pdf_filename))[0]
        pdf_directory = os.path.join(self.save_directory, pdf_basename)
        img_directory = os.path.join(self.save_directory, f"{pdf_basename}/")
        os.makedirs(pdf_directory, exist_ok=True)
        
        # converts to images and provides an output_folder to reduce the memory usage
        try: 
            convert_from_path(pdf_filename, dpi=self.dpi, output_folder=img_directory, fmt="jpeg")
        except:
            print("CONVERSION ERROR: " + pdf_filename)
            pass

        # fixes the names of files output from convert_from_path
        img_files = os.listdir(img_directory)
        for img_file in img_files:
            img_basename = os.path.splitext(os.path.basename(img_file))[0]
            if len(str.split(img_basename, "-")) > 1:
                page_number = int(str.split(img_basename, "-")[-1]) - 1
            else:
                page_number = 0
            output_path = os.path.join(self.save_directory, f"{pdf_basename}/{pdf_basename}_{page_number}.jpg")
            os.rename(os.path.join(img_directory, img_file), output_path)
        return None



    # pdf_directory -> directory to source pdfs
    # save_directory -> directory to save images
    def convert_directory_to_jpegs(self):
        # recursively finds the pdfs in pdf_directory
        pdf_files = []
        for root, _, files in os.walk(self.pdf_directory):
            for filename in files:
                pdf_files.append(os.path.join(root, filename))

        ctx = get_context('spawn')
        with ctx.Pool(processes=30) as pool:
            pool.map(self.convert_pdf_to_jpeg, pdf_files)




