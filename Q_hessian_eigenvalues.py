import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd.functional import hessian

import os
import numpy as np
import matplotlib.pyplot as plt

from model import TwoLayerElementwise
from utils import compute_Q, loss_2norm, get_U_V

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/Q-hessian-eigenvalues'):
    os.mkdir('figures/Q-hessian-eigenvalues')

# Training for one epoch
def train_epoch(epoch, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    output = model.forward(x_train) # 5 x 10
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    U, V = get_U_V(model)
    Q = compute_Q(U, V, activation)
    loss = loss_2norm(U, V, y_train, activation)**2
    return loss, Q


m = 10
n = 5
k = 50
lr = 1e-1
sigma = [0.001, 0.3, 0.6, 0.9]
device = torch.device('cpu')

for activation in ['linear', 'relu', 'tanh', 'sigmoid']: 
    X = torch.eye(n).to(device)
    torch.manual_seed(12345)
    Y = torch.rand(m, n).normal_(mean=0,std=1).to(device)
    fig, ax = plt.subplots()
    
    loss_all = []
    Q_all = []
    imbalance_all = []

    for i in range(len(sigma)):
        torch.manual_seed(12345)
        U = torch.rand(m, k).normal_(mean=0,std=sigma[i]).to(device) # 5 x 50
        torch.manual_seed(12345)
        V = torch.rand(n, k).normal_(mean=0,std=sigma[i]).to(device) # 10 x 50

        model = TwoLayerElementwise(init_U=U, init_V=V, activation=activation)
        
        model.to(device)
        Q = compute_Q(U, V, activation)
        loss = loss_2norm(U, V, Y, activation)**2

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        loss_arr = [loss.detach().cpu().numpy()]
        Q_arr = [Q.detach().cpu().numpy()]
        nepoch = 8000 if activation == 'sigmoid' else 4000
        for epoch in range(nepoch):
            loss, Q_train = train_epoch(epoch, X, Y)
            loss_arr.append(loss.detach().cpu().numpy())
            Q_arr.append(Q_train.detach().cpu().numpy())
        
        Q_all.append(Q_arr)
        loss_all.append(loss_arr)

        U, V = get_U_V(model)
        U_vec = torch.flatten(U)
        V_vec = torch.flatten(V)
        UV_vec = torch.concat((U_vec, V_vec))
        inputs = UV_vec

        def loss_2norm_UV(UV_vec):
            U = torch.reshape(UV_vec[:m*k], (m, k))
            V = torch.reshape(UV_vec[m*k:], (n, k))
            return loss_2norm(U, V, Y, activation)**2

        H = hessian(loss_2norm_UV, inputs)
        eig_vals = torch.linalg.eigvals(H)
        eig_vals = torch.real(eig_vals)

        plt.xlabel('Eigenvalues of Hessian', fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        if np.abs(Q_all[i][0]) > 1e-3:
            Q_label = 'Q={0:.3g}'.format(Q_all[i][0])
        else:
            Q_label = 'Q={0:.2e}'.format(Q_all[i][0])
        bin_vals, bins, patches = plt.hist(eig_vals.detach().cpu().numpy(), bins=np.arange(1e-3, 401, 4, dtype=float), \
                                           alpha=0.8, label=Q_label)

    
    if activation == 'linear':
        plt.xlim(2, 260)
        plt.ylim(0, 21)
        plt.xticks([0, 50, 100, 150, 200, 250], fontsize=18)
        plt.yticks([0, 5, 10, 15, 20], fontsize=18)
    elif activation == 'relu':
        plt.xlim(5, 160)
        plt.ylim(0, 29)
        plt.xticks([0, 50, 100, 150], fontsize=18)
        plt.yticks([0, 5, 10, 15, 20, 25], fontsize=18)
    elif activation == 'tanh':
        plt.xlim(3, 130)
        plt.ylim(0, 20)
        plt.xticks([0, 50, 100], fontsize=18)
        plt.yticks([0, 5, 10, 15], fontsize=18)
    elif activation == 'sigmoid':
        plt.xlim(3, 160)
        plt.ylim(0, 41)
        plt.xticks([0, 50, 100, 150], fontsize=18)
        plt.yticks([0, 10, 20, 30, 40], fontsize=18)
    else:
        raise

    ax.xaxis.offsetText.set_fontsize(18)
    ax.yaxis.offsetText.set_fontsize(18)
    plt.legend(fontsize=16)
    plt.savefig("figures/Q-hessian-eigenvalues/Q-sharpness_{}_mnk_{}_{}_{}.pdf".format(activation, m, n, k), bbox_inches='tight')
