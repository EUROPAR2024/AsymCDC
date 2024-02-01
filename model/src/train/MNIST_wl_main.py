# Code for "i-RevNet: Deep Invertible Networks", ICLR 2018
# Author: Joern-Henrik Jacobsen, 2018
#
# Modified from Pytorch examples code.
# Original license shown below.
# =============================================================================
# BSD 3-Clause License
#
# Copyright (c) 2017, 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import math
import numpy as np

from models.iRevNet import iRevNet


parser = argparse.ArgumentParser(description='Train i-RevNet/RevNet on MNIST')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='i-revnet', type=str, help='model type')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--init_ds', default=0, type=int, help='initial downsampling')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--nBlocks', nargs='+', type=int)
parser.add_argument('--nStrides', nargs='+', type=int)
parser.add_argument('--nChannels', nargs='+', type=int)
parser.add_argument('--bottleneck_mult', default=4, type=int, help='bottleneck multiplier')
parser.add_argument('--resume', default='./checkpoint/MNIST/i-revnet-55.t7', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--train', dest='train', action='store_true',
                    help='train model on training set')
parser.add_argument('--dataset', default='MNIST', type=str, help='dataset')
parser.add_argument('--data_root', default='/home/mzq/wl/dataset', type=str, help="root of dataset")
# parser.add_argument('-i', '--invert', dest='invert', action='store_true',
#                     help='invert samples from validation set')
parser.add_argument('-c', '--cuda', dest="cuda", action='store_true', help="if use cuda or not")


def main():
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if(args.dataset == 'MNIST'):
        trainset = torchvision.datasets.MNIST(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
        nClasses = 10
        in_shape = [1, 28, 28]
    if(args.dataset == 'FashionMNIST'):
        trainset = torchvision.datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
        nClasses = 10
        in_shape = [1, 28, 28]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    def get_model(args):
        if (args.model == 'i-revnet'):
            model = iRevNet(nBlocks=args.nBlocks, nStrides=args.nStrides,
                            nChannels=args.nChannels, nClasses=nClasses,
                            init_ds=args.init_ds, dropout_rate=0.1, affineBN=True,
                            in_shape=in_shape, mult=args.bottleneck_mult)
            fname = 'i-revnet-'+str(sum(args.nBlocks)+1)
        elif (args.model == 'revnet'):
            raise NotImplementedError
        else:
            print('Choose i-revnet or revnet')
            sys.exit(0)
        return model, fname

    model, fname = get_model(args)

    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=(0,))  # range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if use_cuda:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            model = checkpoint['model']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        test(model, testloader, testset, start_epoch, use_cuda, best_acc, args.dataset, fname)
        return

    if args.train:
        print('|  Train Epochs: ' + str(args.epochs))
        print('|  Initial Learning Rate: ' + str(args.lr))

        elapsed_time = 0
        best_acc = 0.
        for epoch in range(1, 1+args.epochs):
            start_time = time.time()

            train(model, trainloader, trainset, epoch, args.epochs, args.batch, args.lr, use_cuda, in_shape)
            best_acc = test(model, testloader, testset, epoch, use_cuda, best_acc, args.dataset, fname)

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

        print('Testing model')
        print('* Test results : Acc@1 = %.2f%%' % (best_acc))





criterion = nn.CrossEntropyLoss()


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def train(model, trainloader, trainset, epoch, num_epochs, batch_size, lr, use_cuda, in_shape):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, learning_rate(lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        out, out_bij = model(inputs)               # Forward Propagation
        loss = criterion(out, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
        train_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.data[0], 100.*correct/total))
        sys.stdout.flush()


def test(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, out_bij = model(inputs)
        print(out_bij.shape)
        loss = criterion(out, targets)

        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
        test_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
                'model': model if use_cuda else model,
                'acc': acc,
                'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+fname+'.t7')
        best_acc = acc
    return best_acc


if __name__ == '__main__':
    main()

