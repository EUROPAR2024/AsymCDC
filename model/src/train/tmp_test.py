import torch
import torch.backends.cudnn as cudnn
from torch import nn

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse

from models.utils_cifar import train, test, std, mean, get_hms
from models.iRevNet import iRevNet

import numpy as np

parser = argparse.ArgumentParser(description='Train i-RevNet/RevNet on Cifar')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='i-revnet', type=str, help='model type')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--init_ds', default=0, type=int, help='initial downsampling')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--nBlocks', nargs='+', type=int)
parser.add_argument('--nStrides', nargs='+', type=int)
parser.add_argument('--nChannels', nargs='+', type=int)
parser.add_argument('--bottleneck_mult', default=4, type=int, help='bottleneck multiplier')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--train', dest='train', action='store_true',
                    help='train model on training set')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--data_root', default='/home/mzq/wl/dataset', type=str, help="root of dataset")
parser.add_argument('-i', '--invert', dest='invert', action='store_true',
                    help='invert samples from validation set')
parser.add_argument('-c', '--cuda', dest="cuda", action='store_true', help="if use cuda or not")
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')

args = parser.parse_args()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean[args.dataset], std[args.dataset]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean[args.dataset], std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=transform_test)
    nClasses = 10
    in_shape = [3, 32, 32]
elif(args.dataset == 'cifar100'):
    trainset = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=False, transform=transform_test)
    nClasses = 100
    in_shape = [3, 32, 32]

trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)

model1 = iRevNet(nBlocks=args.nBlocks, nStrides=args.nStrides,
                nChannels=args.nChannels, nClasses=nClasses,
                init_ds=args.init_ds, dropout_rate=0.1, affineBN=True,
                in_shape=in_shape, mult=args.bottleneck_mult)
fname1 = 'i-revnet-'+str(sum(args.nBlocks)+1)

model2 = iRevNet(nBlocks=[4,4,4], nStrides=args.nStrides,
                nChannels=args.nChannels, nClasses=nClasses,
                init_ds=args.init_ds, dropout_rate=0.1, affineBN=True,
                in_shape=in_shape, mult=args.bottleneck_mult)
fname2 = 'i-revnet-4'

use_cuda = args.cuda and torch.cuda.is_available()

if use_cuda:
    model1.cuda()
    model1 = torch.nn.DataParallel(model1, device_ids=(5,))  # range(torch.cuda.device_count()))
    cudnn.benchmark = True
    
    model2.cuda()
    model2 = torch.nn.DataParallel(model2, device_ids=(5,))  # range(torch.cuda.device_count()))
    cudnn.benchmark = True

if use_cuda:
    checkpoint1 = torch.load('./checkpoint/cifar10/i-revnet-55.t7')
else:
    checkpoint1 = torch.load('./checkpoint/cifar10/i-revnet-55.t7', map_location=torch.device('cpu'))
model1 = checkpoint1['model']

if use_cuda:
    checkpoint2 = torch.load('./checkpoint/cifar10/iRevNet_4_best.pth')
else:
    checkpoint2 = torch.load('./checkpoint/cifar10/iRevNet_4_best.pth', map_location=torch.device('cpu'))
model2 = checkpoint2['model']


data, label = next(iter(trainloader))
if use_cuda:
    data = data.cuda()
    label = label.cuda()
y, yi = model1(data)
x = model2.inverse(yi)
print(data)
print(x)
