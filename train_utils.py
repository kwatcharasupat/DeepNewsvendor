import os
import errno
import shutil
from tqdm import tqdm
import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def discriminative_trainer(model, data_loader, optimizer, criterion):
    torch.cuda.synchronize()
    # print(torch.cuda.memory_allocated())
    # model.eval()
    model.train()
    loss_tracker = AverageMeter()

    for (X, Y) in tqdm(data_loader):

        torch.cuda.empty_cache()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')

        X = X.to(device)
        Y = Y.to(device)

        outputs = model(X)
        loss = criterion(outputs, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tracker.update(loss.item())
    return loss_tracker.avg




def discriminative_evaluate(model, data_loader, criterion):
    torch.cuda.empty_cache()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model.eval()
    loss_tracker = AverageMeter()

    for (X, Y) in tqdm(data_loader):
        X = X.to(device)
        Y = Y.to(device)
        
        outputs = model(X)
        loss = criterion(outputs, Y)
        
        loss_tracker.update(loss.item())

    return loss_tracker.avg



class EarlyStopping:

    # MIT License

    # Copyright (c) 2018 Bjarte Mehus Sunde

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score or np.abs(score - self.best_score) < 1e-4:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
