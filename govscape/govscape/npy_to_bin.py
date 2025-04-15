import numpy as np
import os
import struct

class NpyToBin:
    # pass in bin file
    def __init__(self, bin_file, page_indices):
        self.bin_file = bin_file
        self.page_indices = page_indices

    # only append one npy file to bin
    # not currently adding to page indices
    def convert_npy_to_bin(self, npy_file):
        bin_path = self.bin_file
        file_exists = os.path.exists(bin_path)
        file_header_exists = os.path.getsize(bin_path) >= 8
        data = np.load(npy_file, mmap_mode="r")
        num_points, dimension = data.shape

        with open(bin_path, "r+b" if file_exists else "w+b") as file:
            # create embedding file and write placeholder header
            if not file_exists:
                file.write(struct.pack("i", 0))
                file.write(struct.pack("i", 0))

            file.seek(0)
            total_points = struct.unpack("i", file.read(4))[0]
            dimension = struct.unpack("i", file.read(4))[0]
            file.seek(0, os.SEEK_END)

            for i in range(num_points):
                vector = data[i]                    # loads one row at a time
                file.write(vector.tobytes())        # write raw bytes of vector
        
            file.seek(0)
            file.write(struct.pack("i", total_points + num_points))
            file.write(struct.pack("i", dimension))

    
    # append a pdf directory of npy files to bin
    def convert_pdfdir_to_bin(self, embedding_directory):
        bin_path = self.bin_file
        page_indices_path = self.page_indices
        file_exists = os.path.exists(bin_path)
        if not file_exists:
            with open(bin_path, "w+b") as file:
                file.write(struct.pack("i", 0))
                file.write(struct.pack("i", 0))

        with open(bin_path, "r+b") as file, open(page_indices_path, "a+b") as index_file:
            file.seek(0)
            total_points = struct.unpack("i", file.read(4))[0]
            dimension = struct.unpack("i", file.read(4))[0]
            file.seek(0, os.SEEK_END)

            for page in os.listdir(embedding_directory):
                if page.endswith(".npy"):
                    npy_file = os.path.join(embedding_directory, page)
                    data = np.load(npy_file, mmap_mode="r")
                    data_points, data_dimension = data.shape

                    # writes entire file vs. loading each row in convert_npy_to_bin
                    file.write(data.tobytes())
                    total_points += data_points

                    # 32 bytes for pdf name
                    index_file.write(page[0:32].encode('utf-8'))
                    # 4 bytes for page number
                    index_file.write(struct.pack("i", int(page[33:(len(page) - 4)])))

                    if dimension == 0:
                        dimension = data_dimension
                    elif dimension != data_dimension:
                        raise ValueError("dimension of vector in file does not match dimension of data")
            file.seek(0)
            file.write(struct.pack("i", total_points))
            file.write(struct.pack("i", dimension))
