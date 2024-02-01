import torch
import torch.backends.cudnn as cudnn
import torchaudio

import os
import sys
import time
import argparse

from models.utils_cifar import speech_train, speech_test, get_hms
from models.iRevNet import iRevNet
from dataset.gcommands_dataset import SubsetSC

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
parser.add_argument('--data_root', default='/home/mzq/wl/dataset', type=str, help="root of dataset")
parser.add_argument('-i', '--invert', dest='invert', action='store_true',
                    help='invert samples from validation set')
parser.add_argument('-c', '--cuda', dest="cuda", action='store_true', help="if use cuda or not")
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')


args = parser.parse_args()

#------------------------------------Dataset-------------------------------------
trainset = SubsetSC("training")
testset = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = trainset[0]

labels = sorted(list(set(datapoint[2] for datapoint in trainset)))

new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

in_shape = [1, 400, 400]
nClasses = len(labels)

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=args.workers,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=args.workers,
)

# ---------------------------------Model-------------------------------------
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
    speech_test(model, testloader, testset, start_epoch, use_cuda, best_acc, args.dataset, fname)
    exit(0)

if args.train:
    print('|  Train Epochs: ' + str(args.epochs))
    print('|  Initial Learning Rate: ' + str(args.lr))

    elapsed_time = 0
    best_acc = 0.
    for epoch in range(1, 1+args.epochs):
        start_time = time.time()

        speech_train(model, trainloader, trainset, epoch, args.epochs, args.batch, args.lr, use_cuda, in_shape)
        best_acc = speech_test(model, testloader, testset, epoch, use_cuda, best_acc, args.dataset, fname)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    print('Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))
    exit(0)