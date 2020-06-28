import os
import numpy as np
from matplotlib import cbook

dir_load = "PATH"
dir_save = "PATH"

os.chdir(dir_load)  # change the current directory
# cwd = os.getcwd() # get the current working directory

list_files = os.listdir(dir_load)  # list files in dir
print("Files in %r: %s" % (dir_load, list_files))

for file_name in list_files:
    if ".npy" in file_name:
        print("File ", file_name)
        print("Open HR file")
        # Open the images and convert it to 'numpy.array'
        if "image_5" in file_name:
            hr = np.load(file_name)
            new_name = file_name.replace("IMAGE_NAME_TO_REPLACE", "NEW_NAME")
            print(new_name)
            np.save(dir_save + str(new_name), hr)
            # borrar fichero de la carpeta patches
            os.remove(dir_load+file_name)

    print("Done")

print("DoneAll")