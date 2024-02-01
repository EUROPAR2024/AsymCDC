import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
from models.utils_cifar import parity_train, parity_test, get_hms

from coder.mlp_coder import MLPDecoder
from dataset.ParityDataset import ParityDataset

parser = argparse.ArgumentParser(description='Train i-RevNet/RevNet on Cifar')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--data_root', default="/home/data/wl/", type=str, help="data root")
parser.add_argument('--model', default='mlp', type=str, help='model type')
parser.add_argument('--batch', default=16, type=int, help='batch size')
parser.add_argument('--init_ds', default=0, type=int, help='initial downsampling')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--k', default=4, type=int, help="EC parameter k")
parser.add_argument('--in_dim', default=None, type=int, help="input dimension")
parser.add_argument('--out_dim', default=None, type=int, help="output dimension")
parser.add_argument('--bottleneck_mult', default=4, type=int, help='bottleneck multiplier')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')

args = parser.parse_args()

if(args.dataset == 'cifar10'):
    trainset = ParityDataset(root=args.data_root, train=True)
    testset = ParityDataset(root=args.data_root, train=False)
    nClasses = 10
    in_shape = [3, 32, 32]
elif(args.dataset == 'cifar100'):
    trainset = ParityDataset(root=args.data_root, train=True)
    testset = ParityDataset(root=args.data_root, train=False)
    nClasses = 100
    in_shape = [3, 32, 32]
    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)

def get_model(args):
    if (args.model == 'mlp'):
        model = MLPDecoder(ec_k=args.k, in_dim=args.in_dim, out_dim=args.out_dim)
        fname = 'mlp-decoder-'+str(args.in_dim) + '-' + str(args.out_dim)
    elif (args.model == 'conv'):
        raise NotImplementedError
    else:
        print('Choose mlp or conv')
        sys.exit(0)
    return model, fname

model, fname = get_model(args)

use_cuda = True
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=(0,))  # range(torch.cuda.device_count()))
    cudnn.benchmark = True

# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        model = checkpoint['model']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if args.evaluate:
    parity_test(model, testloader, testset, start_epoch, use_cuda, best_acc, args.dataset, fname)
    exit(0)

print('|  Train Epochs: ' + str(args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))

elapsed_time = 0
best_acc = 0.
for epoch in range(1, 1+args.epochs):
    start_time = time.time()

    parity_train(model, trainloader, trainset, epoch, args.epochs, args.batch, args.lr, use_cuda, in_shape)
    best_acc = parity_test(model, testloader, testset, epoch, use_cuda, best_acc, args.dataset, fname)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

print('Testing model')
print('* Test results : Acc@1 = %.2f%%' % (best_acc))

