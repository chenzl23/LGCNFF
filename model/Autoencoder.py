import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, layer_dims, input_dim):
        '''
        :param layer_dims: dimension of each layer
        :param input_dim: dimension of input
        '''

        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.layer_num = len(layer_dims)

        encoder_dims = layer_dims
        decoder_dims = [layer_dims[i] for i in range(self.layer_num - 1, -1, -1)]

        # init encoder
        for idx, dim in enumerate(encoder_dims):
            if idx == 0:
                self.encoder.add_module("encoder" + str(idx), nn.Linear(input_dim, dim))
            else:
                self.encoder.add_module("encoder" + str(idx), nn.Linear(encoder_dims[idx - 1], dim))
            self.encoder.add_module("activation" + str(idx), nn.Sigmoid())
        
        # init decoder
        for idx, dim in enumerate(decoder_dims):
            if idx == (self.layer_num - 1):
                self.decoder.add_module("decoder" + str(idx), nn.Linear(dim, input_dim))
            else:
                self.decoder.add_module("decoder" + str(idx), nn.Linear(dim, decoder_dims[idx + 1]))
            self.decoder.add_module("activation" + str(idx), nn.ReLU())

    def forward(self, x):
        encode = self.encoder(x)
        self.data_rho = encode.mean(0) 
        decode = self.decoder(encode)
        return encode, decode

    def rho_loss(self, rho, size_average=True):        
        # self.data_rho += 1e-8
        dkl = - rho * torch.log(self.data_rho) - (1-rho)*torch.log(1-self.data_rho) 
        if size_average:
            rho_loss = dkl.mean()
        else:
            rho_loss = dkl.sum()
        return rho_loss
