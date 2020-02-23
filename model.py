import torch

import torch.nn as nn
import numpy as np
import params

def generate_fc(n_hidden, n_nodes):
    layers = []

    in_chan = n_nodes[0]
    for i in range(n_hidden + 2):
        out_chan = n_nodes[i]
        layers += [
            nn.Linear(in_chan, out_chan),
            nn.LeakyReLU(),
            nn.Dropout(params.DROPOUT)
        ]
        in_chan = out_chan

    return nn.Sequential(*layers)

class DeepVendorSimple(nn.Module):

    def __init__(self, model_type = 'simple_fc', n_hidden = 2, n_nodes = [4,3,2,1]):
        super().__init__()

        self.n_features = n_nodes[0]

        if model_type == 'simple_fc':
            self.net = generate_fc(n_hidden, n_nodes)
        else:
            raise NotImplementedError


    def forward(self, x):

        '''
        input   x:      size n_product by n_obs by n_features
        output  y:      size n_product
        '''
        y = self.net(x)
        y = y.squeeze()

        return y


def __sanity_check():
    from hyperband import random_nn_params

    hidden_layers, n_nodes, _, _ = random_nn_params()

    model = DeepVendorSimple(n_hidden = hidden_layers, n_nodes = n_nodes)

    n_product = 10
    n_months = 24
    n_features = n_nodes[0]

    foo = np.random.rand(n_product, n_months, n_features)
    foo = torch.Tensor(foo)

    y = model(foo)

    print(y)
    print(y.shape)


# __sanity_check()