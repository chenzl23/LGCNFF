import torch
import torch.nn as nn
import torch.nn.functional as F

class FCnet(nn.Module):
    def __init__(self, N, input_dim, layer_dims):
            '''
            :param N: num of inputs
            :param input_dim: dimension of input
            :param layer_dims: dimension of layers
            '''
            super(FCnet,self).__init__()

            self.W = nn.Sequential()
            self.H = nn.Parameter(torch.rand(N, input_dim), requires_grad=True)
            for idx, dim in enumerate(layer_dims):
                if idx == 0:
                    self.W.add_module("Linear" + str(idx), nn.Linear(input_dim, dim))
                else:
                    self.W.add_module("Linear" + str(idx), nn.Linear(layer_dims[idx - 1], dim))
                self.W.add_module("activation" + str(idx), nn.ReLU())

    def forward(self):
        return self.W(self.H)