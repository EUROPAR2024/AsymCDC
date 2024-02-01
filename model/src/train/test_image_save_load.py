from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import dataset
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from models.utils_cifar import train, test, std, mean, get_hms
import argparse
from models.iRevNet import iRevNet
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("logs")

parser = argparse.ArgumentParser(description='Generate Cifar with parity')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 or cifar100)')
parser.add_argument('--data_root', default='/home/mzq/wl/dataset', type=str, help="root of dataset")
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--nBlocks', nargs='+', type=int)
parser.add_argument('--nStrides', nargs='+', type=int)
parser.add_argument('--nChannels', nargs='+', type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--k', type=int, default=4, help="code parameter k")
parser.add_argument('--batch', default=64, type=int, help='batch size')
parser.add_argument('--init_ds', default=0, type=int, help='initial downsampling')
parser.add_argument('--bottleneck_mult', default=4, type=int, help='bottleneck multiplier')
parser.add_argument('-c', '--cuda', dest="cuda", action='store_true', help="if use cuda or not")
parser.add_argument("--device", type=int, default=0, help="select the device id")

args = parser.parse_args()

cuda_device_name = "cuda:" + str(args.device)
cuda_device = torch.device(cuda_device_name)

# load cifar dataset
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean[args.dataset], std[args.dataset]),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize(mean[args.dataset], std[args.dataset]),
# ])

if(args.dataset == 'cifar10'):
    trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=transforms.ToTensor())
    nClasses = 10
    in_shape = [3, 32, 32]
elif(args.dataset == 'cifar100'):
    trainset = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=False, transform=transforms.ToTensor())
    nClasses = 100
    in_shape = [3, 32, 32]

trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
testloader = DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

# load model
model = iRevNet(nBlocks=args.nBlocks, nStrides=args.nStrides,
                nChannels=args.nChannels, nClasses=nClasses,
                init_ds=args.init_ds, dropout_rate=0.1, affineBN=True,
                in_shape=in_shape, mult=args.bottleneck_mult)

use_cuda = args.cuda and torch.cuda.is_available()

if use_cuda:
    model.to(cuda_device)
    model = torch.nn.DataParallel(model, device_ids=(args.device,))  # range(torch.cuda.device_count()))

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if use_cuda:
            checkpoint = torch.load(args.resume, map_location=cuda_device)
        else:
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model = checkpoint['model']
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

trans_toPIL = transforms.ToPILImage()
trans_toTensor = transforms.ToTensor()

# for i, (inputs, targets) in enumerate(trainloader):
#     if use_cuda:
#         inputs.cuda()
#         targets.cuda()
    
#     # compute outputs
#     outs, out_bijs = model(inputs)
    
#     # compute parity outputs
#     for i in range(args.batch / args.k):
#         label = torch.zeros([1, out_bijs.shape[1], out_bijs.shape[2], out_bijs.shape[3]]).cuda()
#         inputs_slice = inputs[i * args.k : (i+1) * args.k]
#         if len(inputs_slice) < args.k:
#             break
            
#         for j in range(args.k):
#             label += (1 / args.k) * inputs_slice[i]
        

inputs, target = next(iter(trainloader))
inputs.to(cuda_device)
target.to(cuda_device)
# print("---------input 1---------")
# print(inputs)

out, outbij = model(inputs)
# print("-----------outbij-----------")
# print(outbij)

label = torch.zeros([1, outbij.shape[1], outbij.shape[2], outbij.shape[3]]).to(cuda_device)

for i in range(args.k):
    label[0] += (1 / args.k) * outbij[i]

# print(label[0])

out2 = model.module.inverse(label)
# print("---------out2 1----------")
# print(out2)

# print("----------input 2----------")
# print(inputs)

train_path = "/home/mzq/wl/tmp/train"
for i in range(len(inputs)):
    img_name = str(i) + ".png"
    tmp_input = inputs[i].clone().detach()
    tmp_input = tmp_input.mul(255).byte()
    tmp_input = tmp_input.cpu().numpy().transpose((1,2,0))
    tmp_input = cv.cvtColor(tmp_input, cv.COLOR_RGB2BGR)
    cv.imwrite(os.path.join(train_path, img_name), tmp_input)

val_path = "/home/mzq/wl/tmp/val"
img_name = "1.png"
tmp_out2 = out2[0].clone().detach()
# print(tmp_out2)
tmp_out2 = tmp_out2.mul(255).byte()
tmp_out2 = tmp_out2.cpu().numpy().transpose((1,2,0))
print("--------tmp_out2-------")
tmp_out2 = cv.cvtColor(tmp_out2, cv.COLOR_RGB2BGR)
# print(tmp_out2)
cv.imwrite(os.path.join(val_path, img_name), tmp_out2)
# print("---------tmp_out2 1----------")
# print(tmp_out2)

train_imgs = torch.zeros_like(inputs)
for i in range(len(inputs)):
    img_name = str(i) + ".png"
    train_img = cv.imread(os.path.join(train_path, img_name))
    train_img = cv.cvtColor(train_img, cv.COLOR_BGR2RGB) 
    train_img = trans_toTensor(train_img)
    train_imgs[i] += train_img
    # print("--------train_img " + str(i) + "--------")
    # print(train_imgs[i])

# print("-----------train_imgs 2---------")
# print(train_imgs)

img_name = "1.png"
test_imgs = torch.zeros([1, inputs.shape[1], inputs.shape[2], inputs.shape[3]])
test_img = cv.imread(os.path.join(val_path, img_name))
print("--------test_img--------")
test_img = cv.cvtColor(test_img, cv.COLOR_BGR2RGB) 
test_img = trans_toTensor(test_img)
test_imgs[0] += test_img
# print("-------test_imgs------")
# print(test_imgs)

train_imgs.to(cuda_device)
test_imgs.to(cuda_device)

out3, out3bij = model(train_imgs)
out4, out4bij = model(test_imgs)
# print("----------out3bij----------")
# print(out3bij)


out5 = torch.zeros([1, outbij.shape[1], outbij.shape[2], outbij.shape[3]]).to(cuda_device)
for i in range(args.k):
    out5[0] += (1 / args.k) * out3bij[i]

# print("----------out4bij--------")
# print(out4bij)
# print("------------out5---------")
# print(out5)

flag = True
for i in range(args.k):
    if not train_imgs[i].equal(inputs[i]):
        flag = False
if flag:
    print("True")
else:
    print("False")
print(out3bij.equal(outbij))
print(out5.equal(out4bij))
print(out5.equal(label))
# print(out2.equal(test_imgs))