from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from random import shuffle


class SR_Dataset(Dataset):
    """Generate the dataset."""

    def __init__(self, csv_file, root_dir, transform=None, stand=False, norm=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            stand: standarization
        """
        self.list_of_images = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.stand = stand
        self.norm = norm

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "t1/",
                                self.list_of_images.iloc[idx, 0])
        image_t1 = np.load(img_name)
        image_t2 = np.load(os.path.join(self.root_dir, "t2/",
                                self.list_of_images.iloc[idx, 0]))
        image_t3 = np.load(os.path.join(self.root_dir, "t3/",
                                        self.list_of_images.iloc[idx, 0]))
        image_t4 = np.load(os.path.join(self.root_dir, "t4/",
                                        self.list_of_images.iloc[idx, 0]))
        image_t5 = np.load(os.path.join(self.root_dir, "t5/",
                                        self.list_of_images.iloc[idx, 0]))
                                        
        poss_targets = [image_t1, image_t2, image_t3, image_t4, image_t5] 
        mediana = [image_t1.mean(), image_t2.mean(), image_t3.mean(), image_t4.mean(), image_t5.mean()] 

        for i in range(len(mediana)):
            for j in range(len(mediana)-1):
                if mediana[i] > mediana[j+1]:
                    var = poss_targets[i]
                    poss_targets[i] = poss_targets[j+1]
                    poss_targets[j+1] = var

        target = np.copy(poss_targets[2])                            
              
        image_t1 = np.transpose(image_t1, (1, 2, 0))  # formato HWC
        image_t1 = transform.downscale_local_mean(image_t1, (2, 2, 1))
        image_t1 = np.transpose(image_t1, (2, 0, 1))  # formato CHW
               
        image_t2 = np.transpose(image_t2, (1, 2, 0))  # formato HWC
        image_t2 = transform.downscale_local_mean(image_t2, (2, 2, 1))
        image_t2 = np.transpose(image_t2, (2, 0, 1))  # formato CHW
               
        image_t3 = np.transpose(image_t3, (1, 2, 0))  # formato HWC
        image_t3 = transform.downscale_local_mean(image_t3, (2, 2, 1))
        image_t3 = np.transpose(image_t3, (2, 0, 1))  # formato CHW
               
        image_t4 = np.transpose(image_t4, (1, 2, 0))  # formato HWC
        image_t4 = transform.downscale_local_mean(image_t4, (2, 2, 1))
        image_t4 = np.transpose(image_t4, (2, 0, 1))  # formato CHW
         
        image_t5 = np.transpose(image_t5, (1, 2, 0))  # formato HWC
        image_t5 = transform.downscale_local_mean(image_t5, (2, 2, 1))
        image_t5 = np.transpose(image_t5, (2, 0, 1))  # formato CHW
        
        if self.norm == True:
            image_t1 = image_t1 / 32767
            image_t2 = image_t2 / 32767
            image_t3 = image_t3 / 32767
            image_t4 = image_t4 / 32767
            image_t5 = image_t5 / 32767
            target = target / 32767
               
        if self.stand == True:
            image_t1_mean = image_t1.mean(keepdims=True, axis=(1, 2))
            image_t1_stddev = image_t1.std(keepdims=True, axis=(1, 2))
            image_t1 = (image_t1 - image_t1_mean)/image_t1_stddev
                
            image_t2_mean = image_t2.mean(keepdims=True, axis=(1, 2))
            image_t2_stddev = image_t2.std(keepdims=True, axis=(1, 2))
            image_t2 = (image_t2 - image_t2_mean)/image_t2_stddev
                
            image_t3_mean = image_t3.mean(keepdims=True, axis=(1, 2))
            image_t3_stddev = image_t3.std(keepdims=True, axis=(1, 2))
            image_t3 = (image_t3 - image_t3_mean)/image_t3_stddev
                
            image_t4_mean = image_t4.mean(keepdims=True, axis=(1, 2))
            image_t4_stddev = image_t4.std(keepdims=True, axis=(1, 2))
            image_t4 = (image_t4 - image_t4_mean)/image_t4_stddev
                
            image_t5_mean = image_t5.mean(keepdims=True, axis=(1, 2))
            image_t5_stddev = image_t5.std(keepdims=True, axis=(1, 2))
            image_t5 = (image_t5 - image_t5_mean)/image_t5_stddev
 
            target_mean = target.mean(keepdims=True, axis=(1,2))
            target_stddev = target.std(keepdims=True, axis=(1, 2))
            target = (target - target_mean)/target_stddev
                
        samples = {'image_1': image_t1.astype('float32'), 'image_2': image_t2.astype('float32'), 'image_3': image_t3.astype('float32'), 'image_4': image_t4.astype('float32'),
                   'image_5': image_t5.astype('float32')}
        t = {'target': target.astype('float32')}

        if self.transform != None:
            samples, t = self.transform(samples, t)
            

        return samples, t


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, samples, t):
        image_t1, image_t2, image_t3, image_t4, image_t5, target = samples['image_1'], samples['image_2'], \
                                                                   samples['image_3'], samples['image_4'], \
                                                                   samples['image_5'], t['target']

        # torch image: C X H X W

        samples = {'image_1': torch.from_numpy(image_t1), 'image_2': torch.from_numpy(image_t2),
                   'image_3': torch.from_numpy(image_t3), 'image_4': torch.from_numpy(image_t4),
                   'image_5': torch.from_numpy(image_t5)}
        t = {'target': torch.from_numpy(target)}

        return samples, t
