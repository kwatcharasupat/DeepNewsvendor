import numpy as np 


import params

def random_nodes(hidden_layers):
    

    l_bound = 0.5
    
    if hidden_layers == 2:
        u_bound = [0.,3.,1.]
    elif hidden_layers == 3:
        u_bound = [0.,3.,2.,1.]
    else:
        raise NotImplementedError
        
    n_nodes = []
    n_nodes.append(params.NUM_FEATURES)

    for i in range(1, hidden_layers+1):
        n = int(np.round((np.random.rand()*(u_bound[i]-l_bound) + l_bound) * n_nodes[i-1]))
        n_nodes.append(n)

    n_nodes.append(1)

    n_nodes = np.array(n_nodes).astype(np.int)

    print(f'Generating network with {hidden_layers+2} layers, each with {n_nodes} nodes')

    return n_nodes

def random_nn_params():

    hidden_layers = np.random.randint(params.NN_MIN_LAYER, params.NN_MAX_LAYER+1)
    n_nodes = random_nodes(hidden_layers)
    lr = np.power(10, np.random.rand()*3. - 5.)
    lambd = np.power(10, np.random.rand()*3. - 5.)
    
    return hidden_layers, n_nodes, lr, lambd


def __sanity_check():
    print(random_nn_params())

__sanity_check()



