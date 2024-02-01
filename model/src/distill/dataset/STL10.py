from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = '/home/data/wl/dataset/stl10-data'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class STL10Instance(datasets.STL10):
    """STL10Instance Dataset.
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

def get_STL10_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    STL10
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if is_instance:
        train_set = STL10Instance(root=data_folder,
                                     download=True,
                                     split='train',
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.STL10(root=data_folder,
                                      download=True,
                                      split='train',
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.STL10(root=data_folder,
                                 download=True,
                                 split='test',
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader