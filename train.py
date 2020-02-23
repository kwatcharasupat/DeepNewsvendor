import random
import torch
import numpy as np
import os
from copy import deepcopy
import datetime
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import params

from train_utils import *

model_type = 'simple_fc'  # 'att_context'
id = 'simple_fc'

# Set hyperparams:
missing = False
num_epochs = params.NUM_EPOCHS
batch_size = params.BATCH_SIZE
anneal_factor = params.ANNEALING_FACTOR
patience = params.PATIENCE

seed = 0

# torch.cuda.synchronize()
torch.cuda.empty_cache()

# set random seeds
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Other imports now
from data_utils import SalesDataset
from torch.utils.data import DataLoader

# TRAIN = params.TRAIN
# TEST = params.TEST

train_data = SalesDataset(None, randomize = True)
test_data = SalesDataset(None, randomize = True)

train_loader = DataLoader(train_data, batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size, shuffle = True)
val_loader = test_loader

#------------------#
# Model Definition #
#------------------#

from hyperband import random_nn_params
from model import DeepVendorSimple
from loss_function import EuclideanLoss, CostFunction

for model_idx in tqdm(range(params.NUM_MODELS)):

    hidden_layers, n_nodes, lr, weight_decay = random_nn_params()

    model = DeepVendorSimple(n_hidden = hidden_layers, n_nodes = n_nodes)

    torch.cuda.empty_cache()
    # torch.cuda.memory_summary()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay)

    criterion = EuclideanLoss(params.SHORTAGE_COST, params.HOLDING_COST)
    test_criterion = CostFunction(params.SHORTAGE_COST, params.HOLDING_COST)

    writer_path = os.path.join(id+'_idx_'+str(model_idx))
    #writer = SummaryWriter(writer_path)

    best_model = None
    best_val_loss = 100000.0

    try:
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        for epoch in tqdm(range(num_epochs)):
            torch.cuda.empty_cache()

            # Train model
            loss = discriminative_trainer(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion)
            print(f'epoch: {epoch}, loss: {loss}')

            # log in tensorboard
            #writer.add_scalar('Training/Prediction Loss', loss, epoch)

            # Eval model
            loss = discriminative_evaluate(model, val_loader, test_criterion)
            #writer.add_scalar('Validation/Prediction Cost', loss, epoch)

            if loss < best_val_loss:
                best_val_loss = loss
                best_model = deepcopy(model)

            # Anneal LR
            early_stopping(loss, model)
            if early_stopping.early_stop:  # and epoch > params.MIN_EPOCH:
                print("Early stopping")
                break

    except KeyboardInterrupt:
        print('Stopping training. Now testing')

    # Test the model
    model = best_model
    torch.cuda.empty_cache()
    loss= discriminative_evaluate(model, test_loader, test_criterion)
    print('Test Prediction Cost: ', loss)
            
