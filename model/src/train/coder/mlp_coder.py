from torch import nn

from coder.coder import Coder


class MLPCoder(Coder):
    def __init__(self, num_in, num_out, in_dim, out_dim, layer_sizes_multiplier):
        super().__init__(num_in, num_out, in_dim)
        self.out_dim = out_dim
        nn_modules = nn.ModuleList()
        in_ch = num_in * in_dim
        l = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=2, padding=1)
        nn_modules.append(l)
        l = nn.MaxPool2d(2)
        nn_modules.append(l)
        l = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*2, kernel_size=2, padding=1)
        nn_modules.append(l)
        l = nn.MaxPool2d(2)
        nn_modules.append(l)
        l = nn.Conv2d(in_channels=in_ch*2, out_channels=in_ch*4, kernel_size=2, padding=1)
        nn_modules.append(l)
        l = nn.MaxPool2d(2)
        nn_modules.append(l)
        
        prev_size = in_ch*4
        # for i, size in enumerate(layer_sizes_multiplier):
        #     my_size = 4096 * size
        #     l = nn.Linear(prev_size, my_size)
        #     prev_size = my_size
        #     nn_modules.append(l)
        #     nn_modules.append(nn.ReLU())
        nn_modules.append(nn.Flatten())
        nn_modules.append(nn.Linear(prev_size, prev_size))
        nn_modules.append(nn.ReLU())
        nn_modules.append(nn.Dropout(0.5))
        nn_modules.append(nn.Linear(prev_size, prev_size))
        nn_modules.append(nn.ReLU())
        nn_modules.append(nn.Dropout(0.5))
        nn_modules.append(nn.Linear(prev_size, self.num_out * self.out_dim))
        self.nn = nn.Sequential(*nn_modules)

    def forward(self, in_data):
        # Flatten inputs
        # val = in_data.view(in_data.size(0), -1)
        out = self.nn(in_data)
        return out


class MLPDecoder(MLPCoder):
    def __init__(self, ec_k, in_dim, out_dim):
        num_in = ec_k
        num_out = 1
        layer_sizes_multiplier = [num_in, num_in]
        super().__init__(num_in, num_out, in_dim, out_dim, layer_sizes_multiplier)
