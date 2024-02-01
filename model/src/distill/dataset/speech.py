from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

import argparse
from dataset.gcommand_loader import GCommandLoader


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
        data_folder = '/home/data/wl/dataset/SpeechCommands/gcommands/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class speechInstance(GCommandLoader):
    """speechInstance Dataset.
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

def get_speech_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    speech
    """
    data_folder = get_data_folder()

    if is_instance:
        train_set = speechInstance(data_folder+'train', window_size=.02, window_stride=.01,
                                    window_type='hamming', normalize=True)
        n_data = len(train_set)
    else:
        train_set = GCommandLoader(data_folder+'train', window_size=.02, window_stride=.01,
                                    window_type='hamming', normalize=True)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True)

    test_set = GCommandLoader(data_folder+'test', window_size=.02, window_stride=.01,
                                window_type='hamming', normalize=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             drop_last=True)
    print(len(train_set) + len(test_set))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader