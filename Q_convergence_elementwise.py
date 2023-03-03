import torch
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from model import TwoLayerElementwise
from utils import compute_Q, loss_2norm, get_U_V

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

m = 5
n = 10
k = 50

if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/Q-convergence'):
    os.mkdir('figures/Q-convergence')

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
    loss = loss_2norm(U, V, y_train, activation)
    return loss, Q


for activation in ['linear', 'relu', 'tanh', 'sigmoid']:
    # data generation
    X = torch.eye(n)
    torch.manual_seed(12345)
    Y = torch.rand(m, n).normal_(mean=0,std=1)

    lr = 1e-3
    sigma = [0.01, 0.1, 1] # variance for intial U,V that determines Q
    loss_all = []
    Q_all = []
    imbalance_all = []
    Q_0 = []

    plt.figure()

    for i in range(3):
        torch.manual_seed(12345)
        U = torch.rand(m, k).normal_(mean=0,std=sigma[i]) # 5 x 50
        torch.manual_seed(12345)
        V = torch.rand(n, k).normal_(mean=0,std=sigma[i]) # 10 x 50

        model = TwoLayerElementwise(init_U=U, init_V=V, activation=activation)
        device = torch.device("cpu")
        model.to(device)

        Q = compute_Q(U, V, activation)
        Q_0.append(Q)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        loss_arr = []
        Q_arr = []
        nepoch = 800000 if activation == 'sigmoid' else 400000
        for epoch in range(nepoch):
            loss, Q_train = train_epoch(epoch, X, Y)
            loss_arr.append(loss.detach().numpy())
            Q_arr.append(Q_train.detach().numpy())
        
        Q_all.append(Q_arr)
        loss_all.append(loss_arr)


    # plot loss
    fig, ax = plt.subplots() 
    for i in range(3):
        if activation == 'tanh':
            plt.plot(loss_all[i], label='Q={:.0f}'.format(Q_0[i]), linewidth=2.5)
        else:
            plt.plot(loss_all[i], label='Q={0:.3g}'.format(Q_0[i]), linewidth=2.5)
    plt.yscale('log')
    if activation == 'sigmoid':
        plt.xticks([0, 4e5, 8e5], fontsize=20)
    else:
        plt.xticks([0, 2e5, 4e5], fontsize=20)
    plt.yticks([10e-3, 10e-1, 10e1], fontsize=22)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.g'))
    plt.xlabel('Training steps', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.legend(fontsize=18)
    plt.savefig('figures/Q-convergence/elementwise_{}_convergence.pdf'.format(activation), bbox_inches='tight')
