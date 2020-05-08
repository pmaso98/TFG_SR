from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from SR_Dataset import SR_Dataset, ToTensor
import cv2
import skimage.metrics as skim
import numpy as np
import matplotlib.pyplot as plt
import earthpy.plot as ep
from torchvision import datasets, transforms
import os
import sys
import math


class SR(nn.Module):
    def __init__(self):
        super(SR, self).__init__()  # más adelant puede que lo cambie
        # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # SISRNET
        self.conv1 = nn.Conv3d(5, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.instancenorm1 = nn.InstanceNorm3d(64, affine=True)  # affine=True --> Learnable parameters
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.instancenorm2 = nn.InstanceNorm3d(1, affine=True)
        # FusionNet
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.instancenorm3 = nn.InstanceNorm3d(1, affine=True)

    def forward(self, image_t1, image_t2, image_t3, image_t4,
                image_t5):  # x=[b, 1, depth, height, weight] --> interpolated image

        data_list = []

        image_t1 = self.up(image_t1)
        image_t1 = torch.unsqueeze(image_t1, 1)
        data_list.append(image_t1)
        image_t2 = self.up(image_t2)
        image_t2 = torch.unsqueeze(image_t2, 1)
        data_list.append(image_t2)
        image_t3 = self.up(image_t3)
        image_t3 = torch.unsqueeze(image_t3, 1)
        data_list.append(image_t3)
        image_t4 = self.up(image_t4)
        image_t4 = torch.unsqueeze(image_t4, 1)
        data_list.append(image_t4)
        image_t5 = self.up(image_t5)
        image_t5 = torch.unsqueeze(image_t5, 1)
        data_list.append(image_t5)

        x = torch.cat(data_list, dim=1)
        # print(x.shape)

        images = []

        # SISRNET

        # print(x.shape)
        # x.unsqueeze(1)

        x = self.conv1(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        # images.append(x)

        # Concatenate tensors

        # x = torch.cat(images, dim=2)
        image_interpol = torch.cat(data_list, dim=1)
        # print(image_interpol.shape)
        image_mean = image_interpol.mean(dim=1)
        image_mean = torch.unsqueeze(image_mean, 1)
        # print(image_mean.shape)
        # print(x.shape)

        # FusionNet
        x = self.conv3(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv3(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv3(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv3(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x = self.conv4(x)
        x = self.instancenorm2(x)
        x = F.leaky_relu(x)

        # print(x.shape)

        x += image_mean

        x = x.double()

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (samples, t) in enumerate(train_loader):  # data corresponde a las 5 imagenes LR, target imag HR
        # print(len(train_loader.dataset))
        image_t1 = samples['image_1'].to(device)
        image_t2 = samples['image_2'].to(device)
        image_t3 = samples['image_3'].to(device)
        image_t4 = samples['image_4'].to(device)
        image_t5 = samples['image_5'].to(device)
        target = t['target'].to(device)
        target = torch.unsqueeze(target, 1)
        optimizer.zero_grad()
        output = model(image_t1.float(), image_t2.float(), image_t3.float(), image_t4.float(), image_t5.float())
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and batch_idx * len(samples) <= len(train_loader.dataset):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(samples), len(train_loader.dataset),
                       100. * batch_idx * len(samples) / len(train_loader.dataset), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    psnr = 0
    predicted_image = []
    real_image = []
    with torch.no_grad():
        for samples, t in test_loader:  # potser sense enumerate
            image_t1 = samples['image_1'].to(device)
            image_t2 = samples['image_2'].to(device)
            image_t3 = samples['image_3'].to(device)
            image_t4 = samples['image_4'].to(device)
            image_t5 = samples['image_5'].to(device)
            target = t['target'].to(device)
            target = torch.unsqueeze(target, 1)
            output = model(image_t1.float(), image_t2.float(), image_t3.float(), image_t4.float(), image_t5.float())
            psnr += PSNR.__call__(output, target)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            real = target.numpy()
            pred_batch_size = output.numpy()
            print(pred_batch_size.shape)
            # show_im(real[0], real[1], pred[0], pred[1])
            for i in range(0, args.test_batch_size):
                print(i)
                predicted_image.append(pred_batch_size[i, :, :, :, :])
                real_image.append((real[i, :, :, :, :]))
                predicted_image[i] = predicted_image[i].reshape(4, 128, 128)
                real_image[i] = real_image[i].reshape(4, 128, 128)
                ep.plot_rgb(predicted_image[i], rgb=(2, 1, 0), stretch=True)
                ep.plot_rgb(real_image[i], rgb=(2, 1, 0), stretch=True)

    test_loss /= len(test_loader.dataset)
    psnr /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), PSNR: ({:.2f} db)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), psnr))


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 1]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Deep neural network for Super-resolution of multitemporal '
                                                 'Remote Sensing Images')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = SR_Dataset(csv_file='/Users/pmaso98/Desktop/TFG/Mini_Dataset/train_dataset.csv',
                               root_dir='/Users/pmaso98/Desktop/TFG/Mini_Dataset/', transform=ToTensor())

    test_dataset = SR_Dataset(csv_file='/Users/pmaso98/Desktop/TFG/Mini_Dataset/test_dataset.csv',
                              root_dir='/Users/pmaso98/Desktop/TFG/Mini_Dataset/', transform=ToTensor())

    validation_dataset = SR_Dataset(csv_file='/Users/pmaso98/Desktop/TFG/Mini_Dataset/validation_dataset.csv',
                                    root_dir='/Users/pmaso98/Desktop/TFG/Mini_Dataset/', transform=ToTensor())

    # print(train_dataset.__len__())
    # print(test_dataset.__len__())
    # print(validation_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = SR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "SR_Net.pt")


if __name__ == '__main__':
    main()
