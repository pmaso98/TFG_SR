import os
import numpy as np

import rasterio as rio
from rasterio import plot as rio_plot

import earthpy.plot as ep

import matplotlib
import matplotlib.pyplot as plt

from skimage import exposure


# file_load = '/Users/pmaso98/Desktop/TFG/Imatges_Sentinel/Book1.csv'
from matplotlib import cbook

dir_load = "/Users/pmaso98/Desktop/TFG/Imatges_Sentinel/NEW_YORK_5/"
dir_save = "/Users/pmaso98/Desktop/TFG/Imatges_Sentinel/NEW_YORK_5/"

os.chdir(dir_load)  # change the current directory
# cwd = os.getcwd() # get the current working directory

list_files = os.listdir(dir_load)  # list files in dir
print("Files in %r: %s" % (dir_load, list_files))

# list_files = pd.read_csv("file_load.csv")
print(list_files)

# dir_save = dir_load
if not os.path.isdir(dir_save + "patches/"):
    os.makedirs(dir_save + "patches/")

PATCH_SIZE = 128
overlap = 1
scale = 1
img = 1

for file_name in list_files:
    if ".tiff" in file_name:
        print("File ", file_name)
        print("Open HR file")
        # Open the images and convert it to 'numpy.array'
        hr = rio.open(file_name)
        hr = hr.read()
        # print(hr.shape)
        # maximo = np.amax(hr)
        # hr = hr/maximo
        # hr = exposure.equalize_hist(hr)
        # print(hr)
        # ep.plot_rgb(hr, rgb=(2, 1, 0), stretch=True)
        # plt.show()


# NORMALIZANDO (dIVIDIENDO POR EL MAXIMO de uint16 == 32767)
        # print(hr.dtype)
        hr = hr.astype(np.uint16)
        hr = hr / 32767
        print(hr.shape)

        # creando indices
        num_i = np.floor(hr.shape[1] / PATCH_SIZE)
        num_j = np.floor(hr.shape[2] / PATCH_SIZE)

        indices_i = np.arange(0, hr.shape[1] - PATCH_SIZE, PATCH_SIZE // overlap)
        indices_j = np.arange(0, hr.shape[2] - PATCH_SIZE, PATCH_SIZE // overlap)

        print("Indices i: ", indices_i)
        print("Indices j: ", indices_j)
        pCount = 0
        for ii in indices_i.astype(int):
            for jj in indices_j.astype(int):
                upper_left_i = ii
                upper_left_j = jj
                crop_point = [upper_left_i,
                              upper_left_j,
                              upper_left_i + PATCH_SIZE,
                              upper_left_j + PATCH_SIZE]
                # crop_point_hr = [x * scale for x in crop_point]
                print(crop_point)

                tile_hr = hr[:, crop_point[0]:crop_point[2], crop_point[1]:crop_point[3]]
                print(tile_hr.dtype)
                np.save(dir_save + "patches/" + "image_" + str(img) + "_" + "patch_" + str(pCount), tile_hr)

                print("DONE ", pCount)
                pCount += 1
        print("Done")
        img += 1
        hr = None

    print("DoneAll")
