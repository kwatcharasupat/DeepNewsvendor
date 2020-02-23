import numpy as np 


import params


def random_nn_params():

    n_layers = np.randint(params.NN_MIN_LAYER, params.NN_MAX_LAYER+1)

