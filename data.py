import numpy as np
import random as rand

my_data = np.genfromtxt('final.csv', delimiter=',')

np.savetxt("ruhaha.csv", my_data, delimiter=",")

np.savetxt("ruhaha.csv", my_data, delimiter=",")

my_data = my_data[1:]
start = my_data[:,1][0]
week = {i:(i-start)//7 for i in my_data[:,1]}

mod_data = my_data
mod_data[:,1] = np.array([week[int(x)] for x in my_data[:,1]])

products = [product for product in set(mod_data[:,0])]

n_week = int(max([w for w in week]))
n_prod = len(products)
n_feat = len(mod_data[0]) - 2

features = np.zeros((n_prod, n_week, n_feat))

for r in mod_data:
    features[int(r[0])-1, int(r[1])] = r[2:]

n_test = int(0.2 * n_week)
n_train = int(0.8 * n_week)

train_idx = []
test_idx = []

for i in range(n_week):
    
    if len(train_idx) >= n_train:
        test_idx.append(i)
        continue
    
    if len(test_idx) >= n_test:
        train_idx.append(i)
        continue
    
    if rand.uniform(0,1) < 0.8:
        train_idx.append(i)
    else:
        test_idx.append(i)
        
train_idx = np.array(sorted(train_idx))

print(train_idx)

test_idx = np.array(sorted(test_idx))

print(test_idx)

train_data = features[:, train_idx, :]
test_data = features[:, test_idx, :]

print(f'test: {n_test}, train: {n_train}')
print(f'test: {test_data.shape}, train: {train_data.shape}')


