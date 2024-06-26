"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR, 2018

(c) Joern-Henrik Jacobsen, 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .model_utils import split, merge, injective_pad, psi


class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=4):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.pad = 2 * out_ch - in_ch
        self.stride = stride
        self.inj_pad = injective_pad(self.pad)
        self.psi = psi(stride)
        if self.pad != 0 and stride == 1:
            in_ch = out_ch * 2
            print('')
            print('| Injective iRevNet |')
            print('')
        layers = []
        if not first:
            layers.append(nn.BatchNorm2d(in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_ch//2, int(out_ch//mult), kernel_size=3,
                      stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), int(out_ch//mult),
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), out_ch, kernel_size=3,
                      padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        if self.pad != 0 and self.stride == 1:
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x)
            x1, x2 = split(x)
            x = (x1, x2)
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iRevNet(nn.Module):
    def __init__(self, nBlocks, nStrides, nClasses, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4):
        super(iRevNet, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print('')
        print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3 + 1))
        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)
        self.bn1 = nn.BatchNorm2d(nChannels[-1]*2, momentum=0.9)
        self.linear = nn.Linear(nChannels[-1]*2, nClasses)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        n = self.in_ch//2
        if self.init_ds != 0:
            x = self.init_psi.forward(x)
        out = (x[:, :n, :, :], x[:, n:, :, :])
        for block in self.stack:
            out = block.forward(out)
        out_bij = merge(out[0], out[1])
        out = F.relu(self.bn1(out_bij))
        out = F.avg_pool2d(out, self.ds)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out_bij

    def inverse(self, out_bij):
        """ irevnet inverse """
        out = split(out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        out = merge(out[0],out[1])
        if self.init_ds != 0:
            x = self.init_psi.inverse(out)
        else:
            x = out
        return x

def iRevNet18(**kwargs):
    return iRevNet(
        nBlocks=[18, 18, 18], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
        in_shape=[3, 32, 32], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet9(**kwargs):
    return iRevNet(
        nBlocks=[9, 9, 9], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
        in_shape=[3, 32, 32], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet4(**kwargs):
    return iRevNet(
        nBlocks=[4, 4, 4], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
        in_shape=[3, 32, 32], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet2(**kwargs):
    return iRevNet(
        nBlocks=[2, 2, 2], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
        in_shape=[3, 32, 32], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet1(**kwargs):
    return iRevNet(
        nBlocks=[1, 1, 1], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
        in_shape=[3, 32, 32], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )
    
def iRevNet4_32(**kwargs):
    return iRevNet(
        nBlocks=[4, 4, 4], nStrides=[1, 2, 2], nChannels=[32, 128, 512],
        in_shape=[32, 64, 64], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet16x64(**kwargs):
    return iRevNet(
        nBlocks=[4, 4, 4], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
        in_shape=[16, 64, 64], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=100
    )

def iRevNet48x64(**kwargs):
    return iRevNet(
        nBlocks=[2, 2, 2], nStrides=[1, 2, 2], nChannels=[48, 192, 768],
        in_shape=[48, 64, 64], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=100
    )

def iRevNet64x64(**kwargs):
    return iRevNet(
        nBlocks=[6, 6, 6], nStrides=[1, 2, 2], nChannels=[64, 256, 1024],
        in_shape=[64, 64, 64], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=100
    )

def iRevNet18_32(**kwargs):
    return iRevNet(
        nBlocks=[18, 18, 18], nStrides=[1, 2, 2], nChannels=[32, 128, 512],
        in_shape=[32, 64, 64], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet24_32(**kwargs):
    return iRevNet(
        nBlocks=[24, 24, 24], nStrides=[1, 2, 2], nChannels=[32, 128, 512],
        in_shape=[32, 64, 64], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet18_32_8(**kwargs):
    return iRevNet(
        nBlocks=[18, 18, 18], nStrides=[1, 2, 2], nChannels=[8, 32, 128],
        in_shape=[8, 128, 128], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    ) 

def iRevNet24_32_8(**kwargs):
    return iRevNet(
        nBlocks=[24, 24, 24], nStrides=[1, 2, 2], nChannels=[8, 32, 128],
        in_shape=[8, 128, 128], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    ) 

def iRevNet8x112(**kwargs):
    return iRevNet(
        nBlocks=[18, 18, 18], nStrides=[1, 2, 2], nChannels=[8, 32, 128],
        in_shape=[8, 112, 112], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet32x56(**kwargs):
    return iRevNet(
        nBlocks=[2, 2, 2], nStrides=[1, 2, 2], nChannels=[32, 128, 512],
        in_shape=[32, 56, 56], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet16x56(**kwargs):
    return iRevNet(
        nBlocks=[2, 2, 2], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
        in_shape=[16, 56, 56], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet48x56(**kwargs):
    return iRevNet(
        nBlocks=[2, 2, 2], nStrides=[1, 2, 2], nChannels=[48, 192, 768],
        in_shape=[48, 56, 56], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet64x56(**kwargs):
    return iRevNet(
        nBlocks=[2, 2, 2], nStrides=[1, 2, 2], nChannels=[64, 256, 1024],
        in_shape=[64, 56, 56], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet32x192(**kwargs):
    return iRevNet(
        nBlocks=[4, 4, 4], nStrides=[1, 2, 2], nChannels=[32, 128, 512],
        in_shape=[32, 192, 192], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )

def iRevNet20x320(**kwargs):
    return iRevNet(
        nBlocks=[4, 4, 4], nStrides=[1, 2, 2], nChannels=[20, 80, 320],
        in_shape=[20, 320, 320], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=35
    )

def iRevNet40x320(**kwargs):
    return iRevNet(
        nBlocks=[4, 4, 4], nStrides=[1, 2, 2], nChannels=[40, 160, 640],
        in_shape=[40, 320, 320], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=35
    )

def iRevNet8x384(**kwargs):
    return iRevNet(
        nBlocks=[18, 18, 18], nStrides=[1, 2, 2], nChannels=[8, 32, 128],
        in_shape=[8, 384, 384], init_ds=0, dropout_rate=0.1, 
        affineBN=True, mult=4, nClasses=10
    )


if __name__ == '__main__':
    model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                    nChannels=[24, 96, 384, 1536], nClasses=1000, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[3, 224, 224],
                    mult=4)
    y = model(Variable(torch.randn(1, 3, 224, 224)))
    print(y.size())
