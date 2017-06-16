import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

activation = nn.LeakyReLU

class MLP_G(nn.Module):
    def __init__(self, input_dim, noise_dim, num_chan, hid_dim, ngpu=1):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu
        self.register_buffer('device_id', torch.zeros(1))
        main = nn.Sequential(
            # Z goes into a linear of size: hid_dim
            nn.Linear(noise_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, num_chan * input_dim * input_dim),
            nn.Sigmoid()
        )
        self.main = main
        self.num_chan = num_chan
        self.input_dim = input_dim
        self.noise_dim = noise_dim

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), inputs.size(1))
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        return output.view(output.size(0), self.num_chan, self.input_dim, self.input_dim)


class MLP_D(nn.Module):
    def __init__(self, input_dim, num_chan, hid_dim,out_dim=1, ngpu=1):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu
        self.register_buffer('device_id', torch.zeros(1))
        main = nn.Sequential(
            # Z goes into a linear of size: hid_dim
            nn.Linear(num_chan * input_dim * input_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, out_dim),

        )
        self.main = main
        self.num_chan = num_chan
        self.input_dim = input_dim

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        #output = output.mean(0)
        return output
