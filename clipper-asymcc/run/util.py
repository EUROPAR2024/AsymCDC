import torch

in_dim = {
    "cifar10": [3, 32, 32],
    "cifar100": [3, 32, 32],
    "mnist": [1, 28, 28],
    "fashion": [1, 28, 28],
    "speech": [1, 400, 400],
}

out_dim = {
    "cifar10": [10],
    "cifar100": [100],
    "mnist": [10],
    "fashion": [10],
    "speech": [35],
}

decode_in_dim = {
    "cifar10": [8, 64, 64],
    "cifar100": [8, 64, 64],
    "mnist": [8, 56, 56],
    "fashion": [8, 56, 56],
    "speech": [5, 320, 320]
}

def try_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

def get_flattened_dim(in_dim):
    if isinstance(in_dim, int):
        return in_dim
    elif len(in_dim) == 2:
        return in_dim[-1]
    elif len(in_dim) > 2:
        return in_dim[-1] * in_dim[-2]