
import numpy as np
import params
import torch

from torch.utils.data import Dataset

class SalesDataset(Dataset):
    # Pytorch dataset for OpenMIC
    def __init__(self, npz_path, randomize = True):
        self.randomize = randomize
        if not self.randomize:
            data = np.load(npz_path)
            self.X = data['X']
            self.Y = data['Y']
            self.length = self.X.shape[0]
        else:
            self.length = 1000

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not self.randomize:
            X = self.X[index]
            Y = self.Y[index]
        else:
            X = np.random.rand(100, 24, params.NUM_FEATURES)
            Y = np.random.rand(100, 24)
        
        X = torch.tensor(X, requires_grad=False, dtype=torch.float32)
        Y = torch.tensor(Y.astype(float), requires_grad=False, dtype=torch.float32)

        return X, Y