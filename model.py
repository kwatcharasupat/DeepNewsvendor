import torch

import torch.nn as nn
import numpy as np



class DeepVendorSimple(nn.Module):

    def __init__(self, n_obs, n_product, n_features, model_type = 'simple_fc'):
        super().__init__()

        self.n_obs = n
        self.n_product = m
        self.n_features = p

        if model_type == 'simple_fc':
            self.fc = generate_fc


    def forward(self, x):

        '''
        input   x:      size n_product by n_obs by n_features
        output  y:      size n_product
        '''

        if not np.array_equal(x.shape, [self.n_product, self.n_obs, self.n_features]):
            raise AttributeError

        y = self.fc(x)

        return y



class HyperBand(nn.Module):

    def __init__(self, n, m, p):
        super().__init__()

        self.n_obs = n
        self.n_product = m
        self.n_features = p
        
