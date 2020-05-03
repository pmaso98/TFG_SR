import os
import numpy as np

# file_load = '/Users/pmaso98/Desktop/TFG/Imatges_Sentinel/Book1.csv'
from matplotlib import cbook

dir_load = "/Users/pmaso98/Desktop/TFG/Imatges_Sentinel/EL_CAIRO_5/patches/t5/"
dir_save = "/Users/pmaso98/Desktop/TFG/Imatges_Sentinel/EL_CAIRO_5/patches/t5/"

os.chdir(dir_load)  # change the current directory
# cwd = os.getcwd() # get the current working directory

list_files = os.listdir(dir_load)  # list files in dir
print("Files in %r: %s" % (dir_load, list_files))

for file_name in list_files:
    if ".npy" in file_name:
        print("File ", file_name)
        print("Open HR file")
        # Open the images and convert it to 'numpy.array'
        hr = np.load(file_name)
        np.save(dir_save + str("CAIRO_"+file_name), hr)
        # borrar fichero de la carpeta patches
        os.remove(dir_load+file_name)

    print("Done")

print("DoneAll")