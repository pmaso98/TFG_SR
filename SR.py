from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from SR_Dataset import SR_Dataset, ToTensor
import cv2
import numpy as np
from torchvision import datasets, transforms


class SR(nn.Module):
    def __init__(self):
        super(SR, self).__init__()  # más adelant puede que lo cambie
        # SISRNET
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.instancenorm1 = nn.InstanceNorm3d(64, affine=True)  # affine=True --> Learnable parameters
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.instancenorm2 = nn.InstanceNorm3d(64, affine=True)
        # FusionNet
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv4 = nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, image_t1, image_t2, image_t3, image_t4, image_t5):  # x=[b, 1, depth, height, weight] --> interpolated image
        # print(image_t1.shape)  # [32,5,64,64]
        x = []
        image_t1.unsqueeze(1)
        x1 = torch.reshape(image_t1, (32, 1, 4, 64, 64))  # (32,1,5,64,64)
        x.append(x1)
        image_t2.unsqueeze(1)
        x2 = torch.reshape(image_t2, (32, 1, 4, 64, 64))  # (32,1,5,64,64)
        x.append(x2)
        image_t3.unsqueeze(1)
        x3 = torch.reshape(image_t3, (32, 1, 4, 64, 64))  # (32,1,5,64,64)
        x.append(x3)
        image_t4.unsqueeze(1)
        x4 = torch.reshape(image_t4, (32, 1, 4, 64, 64))  # (32,1,5,64,64)
        x.append(x4)
        image_t5.unsqueeze(1)
        x5 = torch.reshape(image_t5, (32, 1, 4, 64, 64))  # (32,1,5,64,64)
        x.append(x5)
        image_interpol = []
        images = []

        # SISRNET
        for i in range(5):  # range(x) con x el número de imagenes por cada batch_size

            x = x[i]
            x.unsqueeze(1)
            # print(x.shape)

            image_interpol.append(x)

            x = self.conv1(x)
            x = self.instancenorm1(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm2(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm2(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm2(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm2(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm2(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm2(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm2(x)
            x = F.leaky_relu(x)

            images.append(x)

        # Concatenate tensors

        x = torch.cat(images, dim=2)  # (1, 1, 9, H, W)
        image_interpol = torch.cat(image_interpol, dim=2)
        image_mean = image_interpol.mean(dim=2)

        # FusionNet
        x = self.conv3(x)
        x = self.instancenorm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.instancenorm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.instancenorm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.instancenorm2(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = self.instancenorm1(x)
        x = F.leaky_relu(x)

        x += image_mean

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (samples, t) in enumerate(train_loader):  # data corresponde a las 5 imagenes LR, target imag HR
        # data = cv2.resize(data, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        image_t1 = samples['image_1'].to(device)
        image_t2 = samples['image_2'].to(device)
        image_t3 = samples['image_3'].to(device)
        image_t4 = samples['image_4'].to(device)
        image_t5 = samples['image_5'].to(device)
        target = t['target'].to(device)
        optimizer.zero_grad()
        output = model(image_t1.float(), image_t2.float(), image_t3.float(), image_t4.float(), image_t5.float())
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Deep neural network for Super-resolution of multitemporal '
                                                 'Remote Sensing Images')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    # parser.add_argument('--T-in', type=int, default=9, metavar='N',
    #                    help='input number of images (default: 9')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = SR_Dataset(csv_file='/Users/pmaso98/Desktop/TFG/Dataset/Images_t1.csv',
                         root_dir='/Users/pmaso98/Desktop/TFG/Dataset/', transform=ToTensor())

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [11559, 2890])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = SR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "SR_Net.pt")


if __name__ == '__main__':
    main()
