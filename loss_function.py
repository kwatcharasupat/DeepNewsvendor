import torch
import torch.nn as nn
import numpy as np

class EuclideanLoss(nn.Module):

    def __init__(self, c_p, c_h):
        super().__init__()
        self.c_p = c_p
        self.c_h = c_h

    def forward(self, y, d):
        '''
        y:  prediction, size = (n_product, n_obs)
        d:  actual sales, size = (n_product, n_obs)
        '''

        diff = torch.add(y, -d)
        diff = torch.add(torch.mul(torch.max(diff, torch.zeros(1)), self.c_p), torch.mul(torch.max(-diff, torch.zeros(1)), self.c_h))
        diff = torch.norm(diff)
        diff = torch.sum(diff)
        return diff

class CostFunction(nn.Module):

    def __init__(self, c_p, c_h):
        super().__init__()
        self.c_p = c_p
        self.c_h = c_h

    def forward(self, y, d):
        '''
        y:  prediction, size = (n_product, n_obs)
        d:  actual sales, size = (n_product, n_obs)
        '''

        cost = torch.add(y, -d)
        cost = torch.add(torch.mul(torch.max(cost, torch.zeros(1)), self.c_p), torch.mul(torch.max(-cost, torch.zeros(1)), self.c_h))
        cost = torch.sum(cost)

        return cost