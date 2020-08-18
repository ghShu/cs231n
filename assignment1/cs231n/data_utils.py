# import cPickle as pickle 
import _pickle as pickle #for Python 3.x
import numpy as np
import os

# load single batch of cifar
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return (X, Y)

def load_CIFAR10(ROOT):
# """ load all of cifar """
    xs = []
    ys = []
    for b in list(range(1,6)):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del (X, Y)
    (Xte, Yte) = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return (Xtr, Ytr, Xte, Yte)
