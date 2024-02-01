from __future__ import print_function

import os
import argparse
import socket
import time

import torch

from models import model_dict
from models import TeacherModel

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.MNIST import get_MNIST_dataloaders
from dataset.FashionMNIST import get_FashionMNIST_dataloaders
from dataset.STL10 import get_STL10_dataloaders
from dataset.speech import get_speech_dataloaders

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--irev', action='store_true', default="if use iRevNet or not")

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'mnist', 
                                                                            'fashion', 'stl10', 'speech'], help='dataset')

        # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'resnet8_8', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg19_32', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'Shuffle32', 'iRevNet18', 'iRevNet1', 'iRevNet2', 
                                 'iRevNet4', 'iRevNet9', 'iRevNet18_32', 'iRevNet24_32', 'iRevNet18_32_8', 'iRevNet24_32_8',
                                 'iRevNet8x112', 'iRevNet4_32', 'iRevNet32x192', 'iRevNet32x56', 'iRevNet16x56', 'iRevNet20x320'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--path_s', type=str, default=None, help="student model snapshot")
    parser.add_argument('--en', action='store_true', default=False)
    parser.add_argument('--es', action='store_true', default=False)
    parser.add_argument('--et', action='store_true', default=False)
    parser.add_argument('--ee', action='store_true', default=False)
    
    # EC param K
    parser.add_argument('--ec_k', default=4, type=int, help="EC parameter K")

    opt = parser.parse_args()

    opt.model_t = get_teacher_name(opt.path_t)

    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def load_irevnet_teacher(model_path):
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        model = checkpoint['model']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(model_path, checkpoint['epoch']))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        return None
    
def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 10
    elif opt.dataset == 'mnist':
        train_loader, val_loader, n_data = get_MNIST_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 10
    elif opt.dataset == 'fashion':
        train_loader, val_loader, n_data = get_FashionMNIST_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 10
    elif opt.dataset == 'stl10':
        train_loader, val_loader, n_data = get_STL10_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 10
    elif opt.dataset == 'speech':
        train_loader, val_loader, n_data = get_speech_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 35
    else:
        raise NotImplementedError(opt.dataset)

    # model
    if opt.irev:
        teacher = load_irevnet_teacher(opt.path_t)
    else:
        teacher = load_teacher(opt.path_t, n_cls)
    
    if opt.dataset == 'fashion':
        model_n = TeacherModel(teacher, opt.ec_k, 'fashion')
    else:
        model_n = TeacherModel(teacher, opt.ec_k)
    
    model_s = load_irevnet_teacher(opt.path_s)

    # switch to evaluate mode
    model_s.eval()
    
    total_time1 = 0.0 # time for distilled decoder
    total_time2 = 0.0 # time for naive decoder
    total_time3 = 0.0 # time for original model
    total_time4 = 0.0 # time for encode
    total_len1 = 0
    total_len2 = 0
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data = data.float()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            # ========================================
            parity_num = len(data) // opt.ec_k
            parities = torch.zeros((parity_num, data.shape[1], data.shape[2], data.shape[3]))
            for i in range(parity_num):
                parity = torch.zeros_like(data[0])
                for j in range(opt.ec_k):
                    parity += (1 / opt.ec_k) * data[i*opt.ec_k + j]
                parities[i] = parity
            
            # ========================================
            start_t = time.time()
            _, tmpinput = teacher(data)
            end_t = time.time()
            total_time3 += float(end_t - start_t)*1000.0
            
            if torch.cuda.is_available():
                parities = parities.cuda()
            _, ptyinput = teacher(parities)
            
            for i in range(opt.ec_k):
                for j in range(parity_num):
                    if j == 0:
                        input = torch.cat((tmpinput[:i], 
                                        tmpinput[i + 1 : opt.ec_k], 
                                        ptyinput[0].unsqueeze(0)), 
                                        dim=0)
                    else:
                        input = torch.cat((input,
                                        tmpinput[j * opt.ec_k : j * opt.ec_k + i], 
                                        tmpinput[j * opt.ec_k + i + 1 : (j+1) * opt.ec_k], 
                                        ptyinput[j].unsqueeze(0)),
                                          dim=0)
            if opt.es:
                start_t = time.time()
                if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
                    _, _ = model_s(input.reshape((-1, 8*opt.ec_k, 64, 64)))
                elif opt.dataset == 'mnist' or opt.dataset == 'fashion':
                    _, _ = model_s(input.reshape((-1, 8*opt.ec_k, 56, 56)))
                elif opt.dataset == 'stl10':
                    _, _ = model_s(input.reshape((-1, 8*opt.ec_k, 192, 192)))
                elif opt.dataset == 'speech':
                    _, _ = model_s(input.reshape((-1, 5*opt.ec_k, 320, 320)))
                else:
                    _, _ = model_s(input)
                end_t = time.time()
                total_time1 += float(end_t - start_t)*1000.0
            
            if opt.en:
                start_t = time.time()
                _ = model_n(input)
                end_t = time.time()
                total_time2 += float(end_t - start_t)*1000.0
        
            total_len1 += len(input)
            total_len2 += len(data)
            
    if opt.et:
        avg_time3 = total_time3 / total_len2
        print("==============original model============")
        print("Total time costs: {} ms".format(total_time3))
        print("Total number: {}".format(total_len2))
        print("Average time costs: {} ms".format(avg_time3))
        print("\n")
        
    if opt.en:
        avg_time2 = total_time2 / total_len1
        print("=============naive decoder==========")
        print("Total time costs: {} ms".format(total_time2))
        print("Total number: {}".format(total_len1))
        print("Average time costs: {} ms".format(avg_time2))
        print("\n")
    
    if opt.es:
        avg_time1 = total_time1 / total_len1
        print("=============distilled decoder==========")
        print("Total time costs: {} ms".format(total_time1))
        print("Total number: {}".format(total_len1))
        print("Average time costs: {} ms".format(avg_time1))
        print("\n")

if __name__ == '__main__':
    main()
