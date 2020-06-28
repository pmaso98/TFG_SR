from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from SR_Dataset import SR_Dataset, ToTensor
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import earthpy.plot as ep
import skimage.metrics as skm
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os

logs_base_dir = "runs"
os.makedirs(logs_base_dir, exist_ok=True)

tb = SummaryWriter()


class SR(nn.Module):
    def __init__(self):
        super(SR, self).__init__()  
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        # SISRNET
        self.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.instancenorm1 = nn.InstanceNorm2d(64, affine=True, track_running_stats=True)  # affine=True --> Learnable parameters
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.instancenorm2 = nn.InstanceNorm3d(64, affine=True, track_running_stats=True)
        # FusionNet
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv4 = nn.Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.instancenorm3 = nn.InstanceNorm2d(4, affine=True, track_running_stats=True)

    def forward(self, image_t1, image_t2, image_t3, image_t4, image_t5): 

        data_list = []

        image_t1 = torch.unsqueeze(self.up(image_t1), 1)
        data_list.append(image_t1)
        image_t2 = torch.unsqueeze(self.up(image_t2), 1)
        data_list.append(image_t2)
        image_t3 = torch.unsqueeze(self.up(image_t3), 1)
        data_list.append(image_t3)
        image_t4 = torch.unsqueeze(self.up(image_t4), 1)
        data_list.append(image_t4)
        image_t5 = torch.unsqueeze(self.up(image_t5), 1)
        data_list.append(image_t5)

        input = torch.cat(data_list, dim=1)

        images_SISRNET = []

        # SISRNET

        for i in range(input[1].__len__()): # SISRNET INDIVIDUAL PER IMAGE

            x = input[:, i, :, :, :]

            x = self.conv1(x)
            x = self.instancenorm1(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm1(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm1(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm1(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm1(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm1(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm1(x)
            x = F.leaky_relu(x)

            x = self.conv2(x)
            x = self.instancenorm1(x)
            x = F.leaky_relu(x)

            x = torch.unsqueeze(x, 2)

            images_SISRNET.append(x)

        # Concatenate tensors

        x = torch.cat(images_SISRNET, dim=2)

        # Mean of input images
        image_interpol = torch.cat(data_list, dim=1)
        image_mean = image_interpol.mean(dim=1)

        # FusionNet
        f = self.conv3(x)
        f = self.instancenorm2(f)
        f = F.leaky_relu(f)

        f = self.conv3(f)
        f = self.instancenorm2(f)
        f = F.leaky_relu(f)

        f = self.conv3(f)
        f = self.instancenorm2(f)
        f = F.leaky_relu(f)

        f = self.conv3(f)
        f = self.instancenorm2(f)
        f = F.leaky_relu(f)

        f = f.squeeze()

        f = self.conv4(f)
        f = self.instancenorm3(f)
        f = F.leaky_relu(f)

        output = f + image_mean

        return output, f, image_mean



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (samples, t) in enumerate(train_loader): 
        image_t1 = samples['image_1'].to(device)
        image_t2 = samples['image_2'].to(device)
        image_t3 = samples['image_3'].to(device)
        image_t4 = samples['image_4'].to(device)
        image_t5 = samples['image_5'].to(device)
        target = t['target'].to(device)
        optimizer.zero_grad()
        output, fusion, image_mean = model(image_t1, image_t2, image_t3, image_t4, image_t5)
        loss_f = nn.MSELoss()
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and batch_idx * len(samples) <= len(train_loader.dataset):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx * args.batch_size / len(train_loader.dataset), loss.item()))
        
        tb.add_scalar("Loss_train", loss.item(), epoch)

def test(args, model, device, test_loader, epoch):
    model.eval()
    psnr_input = 0
    psnr_output = 0
    ssim_in = 0
    ssim_out = 0
    cnt = 1
    with torch.no_grad():
        for samples, t in test_loader:  
            image_t1 = samples['image_1'].to(device)
            image_t2 = samples['image_2'].to(device)
            image_t3 = samples['image_3'].to(device)
            image_t4 = samples['image_4'].to(device)
            image_t5 = samples['image_5'].to(device)
            target = t['target'].to(device)
            output, fusion, image_mean = model(image_t1, image_t2, image_t3, image_t4, image_t5)
            image_t1 = image_t1.cpu().numpy()
            image_t2 = image_t2.cpu().numpy()
            image_t3 = image_t3.cpu().numpy()
            image_t4 = image_t4.cpu().numpy()
            image_t5 = image_t5.cpu().numpy()
            real = target.cpu().numpy()
            pred = output.cpu().numpy()
            inp = image_mean.cpu().numpy()
            fusion_img = fusion.cpu().numpy()
            for i in range(real.shape[0]):
                psnr_output += skm.peak_signal_noise_ratio(real[i, :, :, :], pred[i, :, :, :], data_range=real[i, :, :, :].max() - real[i, :, :, :].min())
                psnr_input += skm.peak_signal_noise_ratio(real[i, :, :, :], inp[i, :, :, :], data_range=real[i, :, :, :].max() - real[i, :, :, :].min())
                ssim_bands_out = 0
                ssim_bands_in = 0
                for j in range(real.shape[1]):
                    ssim_bands_out += skm.structural_similarity(real[i, j, :, :], pred[i, j, :, :], data_range = real[i, j, :, :].max() - real[i, j, :, :].min())
                    ssim_bands_in += skm.structural_similarity(real[i, j, :, :], inp[i, j, :, :], data_range = real[i, j, :, :].max() - real[i, j, :, :].min())
                
                ssim_out = ssim_out + ssim_bands_out/real.shape[1]
                ssim_in = ssim_in + ssim_bands_in/real.shape[1]

                real_image = real[i, :, :, :].reshape(4, 128, 128)
                predicted_image = pred[i, :, :, :].reshape(4, 128, 128)
                fusion_image = fusion_img[i, :, :, :].reshape(4, 128, 128)
                input_image = inp[i, :, :, :].reshape(4, 128, 128)
                t1 = image_t1[i, :, :, :].reshape(4, 64, 64)              
                t2 = image_t2[i, :, :, :].reshape(4, 64, 64)                
                t3 = image_t3[i, :, :, :].reshape(4, 64, 64)               
                t4 = image_t4[i, :, :, :].reshape(4, 64, 64)               
                t5 = image_t5[i, :, :, :].reshape(4, 64, 64)
                if i == args.batch_size - 1:
                    plot_images(t1, t2, t3, t4, t5, real_image, predicted_image, fusion_image, input_image, cnt, psnr_output/(cnt*args.batch_size), ssim_out/(cnt*args.batch_size), psnr_input/(cnt*args.batch_size), ssim_in/(cnt*args.batch_size)) 
                
            cnt += 1
    
    ssim_in /= len(test_loader.dataset)            
    ssim_out /= len(test_loader.dataset)
    psnr_input /= len(test_loader.dataset)
    psnr_output /= len(test_loader.dataset)

    print('\nTest set: PSNR OUTPUT: ({:.2f} dB), PSNR INPUT: ({:.2f} dB), SSIM OUTPUT: ({:.2f}), SSIM INPUT: ({:.2f})\n'.format(
        psnr_output, psnr_input, ssim_out, ssim_in))
        
def validation(args, model, device, validation_loader, epoch):
    model.eval()
    
    validation_loss = 0
    psnr = 0
    ssim = 0
    
    with torch.no_grad():
        for samples, t in validation_loader:  
            image_t1 = samples['image_1'].to(device)
            image_t2 = samples['image_2'].to(device)
            image_t3 = samples['image_3'].to(device)
            image_t4 = samples['image_4'].to(device)
            image_t5 = samples['image_5'].to(device)
            target = t['target'].to(device)
            output, fusion, image_mean = model(image_t1, image_t2, image_t3, image_t4, image_t5)
            loss_f = nn.MSELoss()
            real = target.cpu().numpy()
            pred = output.cpu().numpy()
            inp = image_mean.cpu().numpy()
            for i in range(real.shape[0]):
                validation_loss += loss_f(output[i, :, :, :], target[i, :, :, :])
                psnr += skm.peak_signal_noise_ratio(real[i, :, :, :], pred[i, :, :, :], data_range=real[i, :, :, :].max() - real[i, :, :, :].min())
                ssim_bands = 0
                for j in range(real.shape[1]):
                    ssim_bands += skm.structural_similarity(real[i, j, :, :], pred[i, j, :, :], data_range = real[i, j, :, :].max() - real[i, j, :, :].min())
                    
                ssim = ssim + ssim_bands/real.shape[1]
                
        tb.add_scalar("Loss_validation", validation_loss/(len(validation_loader.dataset)), epoch)
        tb.add_scalar("SSIM_validation", ssim/(len(validation_loader.dataset)), epoch)
        tb.add_scalar("PSNR_validation", psnr/(len(validation_loader.dataset)), epoch)
       
    validation_loss /= len(validation_loader.dataset)
    psnr /= len(validation_loader.dataset)
    ssim /= len(validation_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, PSNR: ({:.2f} dB), SSIM: ({:.2f})\n'.format(validation_loss, psnr, ssim))
        
        
def plot_images(t1, t2, t3, t4, t5, target, predicted, fusion, input_imag, cnt, psnr_output, ssim_out, psnr_input, ssim_in):
	  
    fig = plt.figure(constrained_layout=True)
    
    gs = fig.add_gridspec(3, 5)
    
    f_ax1 = fig.add_subplot(gs[0, 0])
    f_ax2 = fig.add_subplot(gs[0, 1])
    f_ax3 = fig.add_subplot(gs[0, 2])
    f_ax4 = fig.add_subplot(gs[0, 3])
    f_ax5 = fig.add_subplot(gs[0, 4])
    f_ax6 = fig.add_subplot(gs[1, 0])
    f_ax7 = fig.add_subplot(gs[1, 1])
    f_ax8 = fig.add_subplot(gs[1, 2])
    f_ax9 = fig.add_subplot(gs[1, 3])
    f_ax10 = fig.add_subplot(gs[1, 4])
    f_ax11 = fig.add_subplot(gs[2, 0])
    f_ax12 = fig.add_subplot(gs[2, 1])
    f_ax13 = fig.add_subplot(gs[2, 2])
    f_ax14 = fig.add_subplot(gs[2, 3])
    f_ax15 = fig.add_subplot(gs[2, 4])
    
    
    ep.plot_rgb(t1, rgb=(2, 1, 0), ax=f_ax1, title="Input 1")
    ep.plot_rgb(t2, rgb=(2, 1, 0), ax=f_ax2, title="Input 2")
    ep.plot_rgb(t3, rgb=(2, 1, 0), ax=f_ax3, title="Input 3")
    ep.plot_rgb(t4, rgb=(2, 1, 0), ax=f_ax4, title="Input 4")
    ep.plot_rgb(t5, rgb=(2, 1, 0), ax=f_ax5, title="Input 5")
    ep.plot_rgb(predicted, rgb=(2, 1, 0), ax=f_ax8, title="SR({:.2f}, {:.2f})".format(psnr_output, ssim_out))
    ep.plot_rgb(target, rgb=(2, 1, 0), ax=f_ax6, title="HR(PSNR/SSIM)")
    ep.plot_rgb(target, rgb=(3, 2, 1), ax=f_ax7, title="NIR, R, G")
    ep.plot_rgb(predicted, rgb=(3, 2, 1), ax=f_ax9, title="NIR, R, G") # NIR R G
    ep.plot_rgb(input_imag, rgb=(2, 1, 0), ax=f_ax10, title="B+M({:.2f}, {:.2f})".format(psnr_input, ssim_in))
    ep.plot_rgb(fusion, rgb=(0, 0, 0), ax=f_ax12, title="FNet Red")
    ep.plot_rgb(fusion, rgb=(1, 1, 1), ax=f_ax13, title="FNet Green")  
    ep.plot_rgb(fusion, rgb=(2, 2, 2), ax=f_ax14, title="FNet Blue")  
    ep.plot_rgb(fusion, rgb=(3, 3, 3), ax=f_ax15, title="FNet NIR")
    ep.plot_rgb(fusion, rgb=(2, 1, 0), ax=f_ax11, title="FNet RGB")       
    
    fig.savefig("Results"+str(cnt)+".png")
		
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Deep neural network for Super-resolution of multitemporal '
                                                 'Remote Sensing Images')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
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
    use_cuda = True

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(str(device))

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = SR_Dataset(csv_file='PATH',
                              root_dir='PATH', transform=ToTensor(), stand=True, norm=False)

    test_dataset = SR_Dataset(csv_file='PATH',
                              root_dir='PATH', transform=ToTensor(), stand=True, norm=False)

    validation_dataset = SR_Dataset(csv_file='PATH',
                              root_dir='PATH', transform=ToTensor(), stand=True, norm=False)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True, drop_last=True, **kwargs)

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.test_batch_size, shuffle=True, drop_last=True, **kwargs)

    model = SR().to(device)
    
    model = model.type(dst_type=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)

    
    for epoch in range(1, args.epochs + 1):
        
        train(args, model, device, train_loader, optimizer, epoch)
        
        validation(args, model, device, validation_loader, epoch)
            
        if epoch == 100:
            test(args, model, device, test_loader, epoch)

        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "SR_Net.pt")
        
    tb.close()


if __name__ == '__main__':
    main()
