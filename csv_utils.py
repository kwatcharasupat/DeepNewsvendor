import numpy as np

def csv_to_npz():
    test_data = np.genfromtxt('test.csv' ,dtype=float, delimiter=',',skip_header=0)
    train_data = np.genfromtxt('train.csv' ,dtype=float, delimiter=',',skip_header=0)

    test_x = test_data[:, :-1]
    train_x = train_data[:, :-1]

    test_y = test_data[:, -1]
    train_y = train_data[:, -1]

    np.savez_compressed('test.npz', X=test_x, Y=test_y)
    np.savez_compressed('train.npz', X=train_x, Y=train_y)

def npz_check():
    data = np.load('train.npz')
    print(data['X'].shape)

npz_check()
