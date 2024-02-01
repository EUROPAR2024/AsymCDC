import torch
import torch.nn as nn
import torchvision.transforms as transforms
from util import try_cuda, get_flattened_dim
from torch import Tensor

class Encoder:
    def __init__(self, ec_k, in_dim) -> None:
        self.ec_k = ec_k
        self.in_dim = in_dim

    def __call__(self):
        pass

class LinearEncoder(Encoder):
    def __call__(self, input):
        assert(len(input) == self.ec_k)
        return torch.sum((1/self.ec_k)*input, dim=0)

class ConvEncoder(Encoder):
    def __init__(self, ec_k, in_dim, intermediate_channels_multiplier=3) -> None:
        super().__init__(ec_k, in_dim)
        self.act = nn.ReLU()
        int_channels = intermediate_channels_multiplier * self.ec_k

        self.nn = nn.Sequential(
            nn.Conv2d(in_channels=self.ec_k, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=2*int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*int_channels, out_channels=2*int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*int_channels, out_channels=2*int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*int_channels, out_channels=4*int_channels,
                      kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*int_channels, out_channels=4*int_channels,
                      kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*int_channels, out_channels=2*int_channels,
                      kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=int_channels, out_channels=self.ec_k,
            nn.Conv2d(in_channels=int_channels, out_channels=1,
                      kernel_size=1, stride=1, padding=0, dilation=1)
        )
        
    def __call__(self, input):
        val = input.view(-1, self.ec_k, self.in_dim[1], self.in_dim[2])  #[k, 3, 32, 32] --> [3, k, 32, 32]
        out = self.nn(val) #[3, 1, 32, 32]
        out = out.reshape(self.in_dim)
        return out
    
class ConcatEncoder(Encoder):
    def __init__(self, ec_k, in_dim, type):
        super().__init__(ec_k, in_dim)
        self.type = type
        
        if self.ec_k != 2 and self.ec_k != 4:
            raise Exception(
            "ConcatenationEncoder currently supports values of `ec_k`of 2 or 4.")

        self.original_height = self.in_dim[1]
        self.original_width = self.in_dim[2]

        if (self.original_height % 2 != 0) or (self.original_width % 2 != 0):
            raise Exception(
                    "ConcatenationEncoder requires that image height and "
                    "width be divisible by 2. Image received with shape: "
                    + str(self.in_dim))

        if self.ec_k == 2:
            self.resized_height = self.original_height
            self.resized_width = self.original_width // 2
        else:
            # `ec_k` = 4
            self.resized_height = self.original_height // 2
            self.resized_width = self.original_width // 2
            
        self.trans_rand_crops = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((self.resized_height, self.resized_width)),
            transforms.ToTensor()
        ])
        self.trans_resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resized_height, self.resized_width)),
            transforms.ToTensor()
        ])

    def __call__(self, in_data):
        if self.type == "crop":
            if self.ec_k == 2:
                return torch.cat([self.trans_rand_crops(e) for e in in_data], 2)
            else:
                tmp_img1 = torch.cat([self.trans_rand_crops(e) for e in in_data[:2]], 2)
                tmp_img2 = torch.cat([self.trans_rand_crops(e) for e in in_data[2:]], 2)
                return torch.cat([tmp_img1,tmp_img2], 1)
        else:
            if self.ec_k == 2:
                return torch.cat([self.trans_resize(e) for e in in_data], 2)
            else:
                tmp_img1 = torch.cat([self.trans_resize(e) for e in in_data[:2]], 2)
                tmp_img2 = torch.cat([self.trans_resize(e) for e in in_data[2:]], 2)
                return torch.cat([tmp_img1,tmp_img2], 1)

class MLPEncoder(Encoder):
    def __init__(self, ec_k, in_dim) -> None:
        super().__init__(ec_k, in_dim)
    
        self.inout_dim = get_flattened_dim(in_dim)  #[3, 32, 32] --> [32*32]

        # Set up the feed-forward neural network consisting of two linear
        # (fully-connected) layers and a ReLU activation function.
        self.nn = nn.Sequential(
            nn.Linear(in_features=self.ec_k * self.inout_dim,
                      out_features=self.ec_k * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.ec_k * self.inout_dim,
                      out_features=1 * self.inout_dim)
        )
        
    def __call__(self, input):
        # Flatten inputs
        val = input.view(input.size(1), -1)  #[k, 3, 32, 32] --> [3, k*32*32]
        
        # Perform inference over encoder model
        # The MLP encoder operates over different channels of input images independently.
        out_list = []
        for _,data in enumerate(val):
            out_list.append(self.nn(data))
        out = torch.stack(out_list, dim=0)
        out = out.view(self.in_dim)
        return out