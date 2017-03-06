from sklearn import linear_model, datasets, metrics
import numpy as np


def gibbs_sample_h(bias_h, theta,x):
    h = []
    z = bias_h + theta.dot(x)
    phv = 1 / (1 + np.exp(-z))
    hu = np.random.random_sample(phv.shape[0])
    for i, hv in enumerate(phv):
        if hv > hu[i]:
            h.append(1)
        else:
            h.append(0)
    return h

def gibbs_sample_v(biash_v, theta, sigma, h):
    v  = []
    z = biash_v + theta.T.dot(h)
    pvh = 1/(1+np.exp(-z))
    for i, vh in enumerate(pvh):
        sp = np.random.normal(vh, sigma[i])
        v.append(sp)
    return v


def train(X,Y):
    sigma = np.std(X,axis=0) + 0.00001
    theta = np.random.randn(hide_nodes, X[0].shape[0])
    bias_v = np.random.random_sample(X[0].shape[0])
    bias_h = np.random.random_sample(hide_nodes)
    for i in range(0, 2):
        for j, x in enumerate(X):
            h = gibbs_sample_h(bias_h, theta, x)
            v = gibbs_sample_v(bias_v,theta,sigma,h)
            print("ss")



if __name__ == "__main__":
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    Y = digits.target
    hide_nodes = 128
    train(X,Y)