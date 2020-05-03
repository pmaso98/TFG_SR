import os
import numpy as np

# file_load = '/Users/pmaso98/Desktop/TFG/Imatges_Sentinel/Book1.csv'
from matplotlib import cbook

dir_load = "/Users/pmaso98/Desktop/TFG/Imatges_Sentinel/NEW_YORK_5/patches/"
dir_save = "/Users/pmaso98/Desktop/TFG/Imatges_Sentinel/NEW_YORK_5/patches/"

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
        if "image_1" in file_name:
            if not os.path.isdir(dir_save + "t1/"):
                os.makedirs(dir_save + "t1/")
            np.save(dir_save + "t1/" + str(file_name), hr)
        elif "image_2" in file_name:
            if not os.path.isdir(dir_save + "t2/"):
                os.makedirs(dir_save + "t2/")
            np.save(dir_save + "t2/" + str(file_name), hr)
        elif "image_3" in file_name:
            if not os.path.isdir(dir_save + "t3/"):
                os.makedirs(dir_save + "t3/")
            np.save(dir_save + "t3/" + str(file_name), hr)
        elif "image_4" in file_name:
            if not os.path.isdir(dir_save + "t4/"):
                os.makedirs(dir_save + "t4/")
            np.save(dir_save + "t4/" + str(file_name), hr)
        elif "image_5" in file_name:
            if not os.path.isdir(dir_save + "t5/"):
                os.makedirs(dir_save + "t5/")
            np.save(dir_save + "t5/" + str(file_name), hr)

        # borrar fichero de la carpeta patches
        os.remove(dir_load+file_name)

    print("Done")

print("DoneAll")




