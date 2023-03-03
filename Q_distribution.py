import torch
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_Q

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/Q-distribution'):
    os.mkdir('figures/Q-distribution')

# plot distribution of Q for 2-layer linear NN with different nonlinearities
m = 100
h = 100
n = 100

for activation in ['linear', 'relu', 'tanh', 'sigmoid']:
    Q = []
    for i in range(1000):
        torch.manual_seed(i**2)
        U = torch.rand(m, h).normal_(mean=0,std=np.sqrt(1/h))
        V = torch.rand(n, h).normal_(mean=0,std=np.sqrt(1/n))
        Q.append(compute_Q(U, V, activation))

    plt.figure()
    plt.xlabel(r'$Q$', fontsize=20)
    plt.ylabel('frequency', fontsize=20)
    counts, bins = np.histogram(Q)
    x = (bins[1:] + bins[:-1]) / 2
    plt.bar(x, counts, width=bins[1]-bins[0])
    plt.xticks(fontsize=18)
    if activation == 'sigmoid':
        plt.xticks([-10050, -10000, -9950], fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('figures/Q-distribution/{}_mhn_100_100_100.pdf'.format(activation), bbox_inches='tight')


# plot distribution of Q for 2-layer linear NN with different layer dimensions
mhn = [(100, 200, 100), (200, 100, 100)] # param dimensions
activation = 'linear'
for (m, h, n) in mhn:
    Q = []
    for i in range(1000):
        torch.manual_seed(i**2)
        U = torch.rand(m, h).normal_(mean=0,std=np.sqrt(1/h))
        V = torch.rand(n, h).normal_(mean=0,std=np.sqrt(1/n))
        Q.append(compute_Q(U, V, activation))

    plt.figure()
    plt.xlabel(r'$Q$', fontsize=20)
    plt.ylabel('frequency', fontsize=20)
    counts, bins = np.histogram(Q)
    x = (bins[1:] + bins[:-1]) / 2
    plt.bar(x, counts, width=bins[1]-bins[0])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig('figures/Q-distribution/{}_mhn_{}_{}_{}.pdf'.format(activation, m, h, n), bbox_inches='tight')
