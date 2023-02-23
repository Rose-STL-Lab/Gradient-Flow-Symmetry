import torch
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from model import TwoLayerElementwise

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

for activation in ['linear', 'relu', 'tanh', 'sigmoid']:
    # data generation
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


    lr = 1e-3
    sigma = [0.01, 0.1, 1]
    loss_all = []
    Q_all = []
    imbalance_all = []
    Q_0 = []

    for i in range(3):
        torch.manual_seed(12345)
        U = torch.rand(m, k).normal_(mean=0,std=sigma[i]) # 5 x 50
        torch.manual_seed(12345)
        V = torch.rand(n, k).normal_(mean=0,std=sigma[i]) # 10 x 50

        model = TwoLayerElementwise(init_U=U, init_V=V, activation=activation)
        device = torch.device("cpu")
        model.to(device)

        if activation == 'linear':
            Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(V.T, V)) #linear
        elif activation == 'relu':
            Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(relu(V.T), relu(V))) #relu
        elif activation == 'tanh':
            Q = torch.trace(torch.matmul(U.T, U) - torch.matmul(torch.cosh(V.T), torch.cosh(V))) #tanh
        elif activation == 'sigmoid':
            Q = 0.5 * torch.trace(torch.matmul(U.T, U)) - torch.sum(V) - torch.sum(torch.exp(V)) #sigmoid
        else:
            raise 

        Q_0.append(Q)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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


    # plot loss
    fig, ax = plt.subplots() 
    for i in range(3):
        if activation == 'tanh':
            plt.plot(loss_all[i], label='Q={:.0f}'.format(Q_0[i]), linewidth=2.5)
        else:
            plt.plot(loss_all[i], label='Q={0:.3g}'.format(Q_0[i]), linewidth=2.5)
    plt.yscale('log')
    # plt.xscale('log')
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

