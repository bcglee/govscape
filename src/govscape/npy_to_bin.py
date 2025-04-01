import numpy as np
import os
import struct

class NpyToBin:
    # pass in bin file
    def __init__(self, bin_file):
        self.bin_file = bin_file
        self.page_indices = []

    # only append one npy file to bin
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
                vector = data[i]                    # This loads just one row at a time
                file.write(vector.tobytes())          # Write the raw bytes of the vector
        
            file.seek(0)
            file.write(struct.pack("i", total_points + num_points))
            file.write(struct.pack("i", dimension))
            # self.page_indicies.append()
        # NEED TO CHANGE PAGE INDEX LOGIC write to bin file?
        # self.page_indices.append(self.total_vectors)

    
    # append a pdf directory of npy files to bin
    def convert_pdfdir_to_bin(self, embedding_directory):
        bin_path = self.bin_file
        file_exists = os.path.exists(bin_path)
        file_header_exists = os.path.getsize(bin_path) >= 8

        with open(bin_path, "r+b") as file:
            total_points = 0
            dimension = 0
            if file_exists and file_header_exists:
                file.seek(0)
                total_points = struct.unpack("i", file.read(4))[0]
                dimension = struct.unpack("i", file.read(4))[0]
                file.seek(0, os.SEEK_END)
            else :
                # add placeholder header first if bin file doesn't exist, will update later
                file.write(struct.pack("i", total_points))
                file.write(struct.pack("i", dimension))
            
            # curr_index = num_points
                # append current index to mark start of page
                # self.page_indices.append(curr_index)
                # loop through pages of pdf
            for page in os.listdir(embedding_directory):
                if page.endswith(".npy"):
                    npy_file = os.path.join(embedding_directory, page)
                    data = np.load(npy_file, mmap_mode="r")
                    data_points, data_dimension = data.shape
                    # writes entire file vs. loading each row in convert_npy_to_bin
                    file.write(data.tobytes())
                    total_points += data_points
                    if dimension == 0:
                        dimension = data_dimension
                    elif dimension != data_dimension:
                        raise ValueError("dimension of vector in file does not match dimension of data")
            file.seek(0)
            file.write(struct.pack("i", total_points))
            file.write(struct.pack("i", dimension))

        # NEED TO CHANGE PAGE INDEX LOGIC
        # page_indices_path = os.path.join(os.path.dirname(bin_path), "page_indices.npy")
        # np.save(page_indices_path, page_indices) 
