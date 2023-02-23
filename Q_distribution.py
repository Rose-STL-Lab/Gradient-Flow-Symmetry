import torch
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/Q-distribution'):
    os.mkdir('figures/Q-distribution')

def compute_Q(U, V, activation):
    Q_imbalance = torch.trace(torch.matmul(U.T, U) - torch.matmul(V, V.T))
    if activation == 'linear':
        Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(V, V.T)) #linear
        # norm_Y_X = torch.norm(y_train - torch.matmul(U, V.T))
    elif activation == 'relu':
        Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(relu(V), relu(V.T))) #relu
        # norm_Y_X = torch.norm(y_train - torch.matmul(U, relu(V.T)))
    elif activation == 'tanh':
        Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(torch.cosh(V), torch.cosh(V.T))) #tanh
        # norm_Y_X = torch.norm(y_train - torch.matmul(U, tanh(V.T)))
    elif activation == 'sigmoid':
        Q = 0.5 * torch.trace(torch.matmul(U.T, U)) - torch.sum(V) - torch.sum(torch.exp(V)) #sigmoid
        # norm_Y_X = torch.norm(y_train - torch.matmul(U, sigmoid(V.T)))
    else:
        raise 
    return Q

# param dimension
m = 100
h = 100
n = 100

activation = 'sigmoid' # linear/relu/tanh/sigmoid
for activation in ['linear', 'relu', 'tanh', 'sigmoid']:
    Q = []
    for i in range(1000):
        torch.manual_seed(i**2)
        U = torch.rand(m, h).normal_(mean=0,std=np.sqrt(1/h)) # 5 x 50
        V = torch.rand(h, n).normal_(mean=0,std=np.sqrt(1/n)) # 10 x 50
        Q.append(compute_Q(U, V, activation))

    plt.figure()
    plt.xlabel(r'$Q$', fontsize=20)
    plt.ylabel('frequency', fontsize=20)
    # plt.hist(Q, bins=10)
    counts, bins = np.histogram(Q)
    x = (bins[1:] + bins[:-1]) / 2
    plt.bar(x, counts, width=bins[1]-bins[0])
    plt.xticks(fontsize=18)
    if activation == 'sigmoid':
        plt.xticks([-10050, -10000, -9950], fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig('figures/Q-distribution/{}_mhn_100_100_100.pdf'.format(activation), bbox_inches='tight')


# param dimension
mhn = [(100, 200, 100), (200, 100, 100)]

activation = 'linear'
for (m, h, n) in mhn:
    Q = []
    for i in range(1000):
        torch.manual_seed(i**2)
        U = torch.rand(m, h).normal_(mean=0,std=np.sqrt(1/h)) # 5 x 50
        V = torch.rand(h, n).normal_(mean=0,std=np.sqrt(1/n)) # 10 x 50
        Q.append(compute_Q(U, V, activation))

    plt.figure()
    plt.xlabel(r'$Q$', fontsize=20)
    plt.ylabel('frequency', fontsize=20)
    # plt.hist(Q, bins=10)
    counts, bins = np.histogram(Q)
    x = (bins[1:] + bins[:-1]) / 2
    plt.bar(x, counts, width=bins[1]-bins[0])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig('figures/Q-distribution/{}_mhn_{}_{}_{}.pdf'.format(activation, m, h, n), bbox_inches='tight')
