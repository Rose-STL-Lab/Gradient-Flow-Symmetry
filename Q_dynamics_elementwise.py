import torch
from torch import nn
import os
import numpy as np
from model import TwoLayerElementwise
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

# data generation
m = 5
n = 10
k = 50

if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/Q-dynamics'):
    os.mkdir('figures/Q-dynamics')


for activation in ['linear', 'relu', 'tanh', 'sigmoid']:

    X = torch.eye(n)
    torch.manual_seed(12345)
    Y = torch.rand(m, n).normal_(mean=0,std=1)
    plt.figure()

    # Training for one epoch
    def train_epoch(epoch, x_train, y_train):
        model.train()
        optimizer.zero_grad()
        output = model.forward(x_train) # 5 x 10
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        param_list = []
        for param in model.parameters():
        	  param_list.append(param.data)
        U = param_list[0].clone() # 5 x 50
        V = param_list[1].clone() # 10 x 50
        Q_imbalance = torch.trace(torch.matmul(U.T, U) - torch.matmul(V.T, V))

        if activation == 'linear':
            Q = Q_imbalance #linear
            norm_Y_X = torch.norm(y_train - torch.matmul(U, V.T))
        elif activation == 'relu':
            Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(relu(V.T), relu(V))) #relu
            norm_Y_X = torch.norm(y_train - torch.matmul(U, relu(V.T)))
        elif activation == 'tanh':
            Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(torch.cosh(V.T), torch.cosh(V))) #tanh
            norm_Y_X = torch.norm(y_train - torch.matmul(U, tanh(V.T)))
        elif activation == 'sigmoid':
            Q = 0.5 * torch.trace(torch.matmul(U.T, U)) - torch.sum(V) - torch.sum(torch.exp(V)) #sigmoid
            norm_Y_X = torch.norm(y_train - torch.matmul(U, sigmoid(V.T)))
        else:
            raise 

        return norm_Y_X, Q, Q_imbalance


    sigma = 0.1
    lr = [0.1, 0.01, 0.001]
    loss_all = []
    Q_all = []
    imbalance_all = []
    UU_plus_VV_0 = 0

    for i in range(3):
        torch.manual_seed(12345)
        U = torch.rand(m, k).normal_(mean=0,std=sigma) # 5 x 50
        torch.manual_seed(12345)
        V = torch.rand(n, k).normal_(mean=0,std=sigma) # 10 x 50

        model = TwoLayerElementwise(init_U=U, init_V=V, activation=activation)
        device = torch.device("cpu")
        model.to(device)

        if activation == 'linear':
            Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(V.T, V)) #linear
            UU_plus_VV_0 = torch.trace(torch.matmul(U.T, U) + torch.matmul(V.T, V))
        elif activation == 'relu':
            Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(relu(V.T), relu(V))) #relu
            UU_plus_VV_0 = torch.trace(torch.matmul(U.T, U) + torch.matmul(relu(V.T), relu(V)))
        elif activation == 'tanh':
            Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(torch.cosh(V.T), torch.cosh(V))) #tanh
            UU_plus_VV_0 = torch.trace(torch.matmul(U.T, U) + torch.matmul(torch.cosh(V.T), torch.cosh(V)))
        elif activation == 'sigmoid':
            Q = 0.5 * torch.trace(torch.matmul(U.T, U)) - torch.sum(V) - torch.sum(torch.exp(V)) #sigmoid
            UU_plus_VV_0 = 0.5 * torch.trace(torch.matmul(U.T, U)) + torch.sum(V) + torch.sum(torch.exp(V))
        else:
            raise 

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr[i])

        loss_arr = []
        Q_arr = []
        Q_arr_imbalance = []
        nepoch = 800000 if activation == 'sigmoid' else 400000
        for epoch in range(nepoch):
            loss, Q_train, Q_train_imbalance = train_epoch(epoch, X, Y)
            # loss_arr.append((loss / torch.norm(Y)).detach().numpy())
            loss_arr.append(loss.detach().numpy())
            Q_arr.append(Q_train.detach().numpy())
            Q_arr_imbalance.append(Q_train_imbalance.detach().numpy())
        
        Q_all.append(Q_arr)
        imbalance_all.append(Q_arr_imbalance)
        loss_all.append(loss_arr)

    # plot loss and Q together
    plt.figure()  
    fig, ax = plt.subplots(2, sharex='col', sharey='row')
    for i in range(3):
        dQ = np.abs(Q_all[i]) / UU_plus_VV_0
        dQ = dQ - dQ[0]
        ax[0].plot(dQ, linewidth=2.5)
        ax[1].plot(loss_all[i], label='lr={}'.format(lr[i]), linewidth=2.5)

    if activation == 'sigmoid':
        ax[0].set_xticks([0, 4e5, 8e5], fontsize=18)
    else:
        ax[0].set_xticks([0, 2e5, 4e5], fontsize=18)
    if activation == 'linear':
        ax[0].set_yticks([0, 2.00e-4, 4.00e-4, 6.00e-4], fontsize=18)

    ax[0].set_yticklabels([0, 2.00e-4, 4.00e-4, 6.00e-4], fontsize=18)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax[1].set_yticklabels([10e-3, 10e-1, 10e1], fontsize=18)
    # ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.g'))
    ax[1].set_xticklabels([0, 2e5, 4e5], fontsize=18)
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1g'))


    ax[0].set_ylabel(r'$\Delta\tilde{Q}$', fontsize=20)
    ax[1].set_xlabel('Training steps', fontsize=22)
    ax[1].set_ylabel('Loss', fontsize=22)
    # ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.legend(fontsize=15)
    plt.savefig('figures/Q-dynamics/elementwise_{}_loss_Q.pdf'.format(activation), bbox_inches='tight')
