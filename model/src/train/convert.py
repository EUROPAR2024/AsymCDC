import torch

import argparse

parser = argparse.ArgumentParser(description='Train i-RevNet/RevNet on Cifar')
parser.add_argument('--path', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()

checkpoint = torch.load(args.path)
start_epoch = checkpoint['epoch']
best_acc = checkpoint['acc']
model = checkpoint['model']

state = {
        'model': model if torch.cuda.is_available() else model,
        'acc': best_acc,
        'epoch': start_epoch,
}

torch.save(state, args.path, _use_new_zipfile_serialization=False)