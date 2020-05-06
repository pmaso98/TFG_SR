from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SR_Dataset(Dataset):
    """Generate the dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.list_of_images = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "t1/",
                                self.list_of_images.iloc[idx, 0])
        image_t1 = np.load(img_name)
        image_t1 = np.transpose(image_t1, (1, 2, 0))  # formato HWC
        image_t1 = transform.downscale_local_mean(image_t1, (2, 2, 1))
        image_t1 = np.transpose(image_t1, (2, 0, 1))  # formato CHW
        image_t2 = None
        image_t3 = None
        image_t4 = None
        image_t5 = None
        target = None
        for i in os.listdir(os.path.join(self.root_dir, "t2/")):
            if self.list_of_images.iloc[idx, 0] == i:
                img_name2 = os.path.join(self.root_dir, "t2/", i)
                image_t2 = np.load(img_name2)
                image_t2 = np.transpose(image_t2, (1, 2, 0))  # formato HWC
                image_t2 = transform.downscale_local_mean(image_t2, (2, 2, 1))
                image_t2 = np.transpose(image_t2, (2, 0, 1))  # formato CHW
                # print(image_t2.shape)
        for i in os.listdir(os.path.join(self.root_dir, "t3/")):
            if self.list_of_images.iloc[idx, 0] == i:
                img_name3 = os.path.join(self.root_dir, "t3/", i)
                image_t3 = np.load(img_name3)
                target = image_t3
                image_t3 = np.transpose(image_t3, (1, 2, 0))  # formato HWC
                image_t3 = transform.downscale_local_mean(image_t3, (2, 2, 1))
                image_t3 = np.transpose(image_t3, (2, 0, 1))  # formato CHW
                # print(image_t3.shape)
        for i in os.listdir(os.path.join(self.root_dir, "t4/")):
            if self.list_of_images.iloc[idx, 0] == i:
                img_name4 = os.path.join(self.root_dir, "t4/", i)
                image_t4 = np.load(img_name4)
                image_t4 = np.transpose(image_t4, (1, 2, 0))  # formato HWC
                image_t4 = transform.downscale_local_mean(image_t4, (2, 2, 1))
                image_t4 = np.transpose(image_t4, (2, 0, 1))  # formato CHW
                # print(image_t4.shape)
        for i in os.listdir(os.path.join(self.root_dir, "t5/")):
            if self.list_of_images.iloc[idx, 0] == i:
                img_name5 = os.path.join(self.root_dir, "t5/", i)
                image_t5 = np.load(img_name5)
                image_t5 = np.transpose(image_t5, (1, 2, 0))  # formato HWC
                image_t5 = transform.downscale_local_mean(image_t5, (2, 2, 1))
                image_t5 = np.transpose(image_t5, (2, 0, 1))  # formato CHW
                # print(image_t5.shape)

        samples = {'image_1': image_t1, 'image_2': image_t2, 'image_3': image_t3, 'image_4': image_t4,
                   'image_5': image_t5}
        t = {'target': target}
        # print(target.shape)

        if self.transform:
            samples, t = self.transform(samples, t)

        return samples, t


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, samples, t):
        image_t1, image_t2, image_t3, image_t4, image_t5, target = samples['image_1'], samples['image_2'], \
                                                                   samples['image_3'], samples['image_4'], \
                                                                   samples['image_5'], t['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_t1 = image_t1.transpose((2, 0, 1))
        image_t2 = image_t2.transpose((2, 0, 1))
        image_t3 = image_t3.transpose((2, 0, 1))
        image_t4 = image_t4.transpose((2, 0, 1))
        image_t5 = image_t5.transpose((2, 0, 1))
        target = target.transpose((2, 0, 1))
        samples = {'image_1': torch.from_numpy(image_t1), 'image_2': torch.from_numpy(image_t2),
                   'image_3': torch.from_numpy(image_t3), 'image_4': torch.from_numpy(image_t4),
                   'image_5': torch.from_numpy(image_t5)}
        t = {'target': torch.from_numpy(target)}

        return samples, t
