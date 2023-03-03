import torch
from torch import nn
import os
import numpy as np
from model import TwoLayerRadial
import matplotlib.pyplot as plt

from utils import get_U_V


# data generation
m = 5
n = 10
k = 5

if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/Q-convergence'):
    os.mkdir('figures/Q-convergence')
if not os.path.exists('cache'):
    os.mkdir('cache')
if not os.path.exists('cache/train_radial'):
    os.mkdir('cache/train_radial')

# Training for one epoch
def train_epoch(epoch, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    output = model.forward(x_train) # 5 x 10
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    U, V = get_U_V(model)
    Q = torch.trace(torch.matmul(U, U.T)) + torch.trace(torch.matmul(V, V.T))
    return loss, Q

X = torch.eye(n)
torch.manual_seed(12345)
Y = torch.rand(m, n).normal_(mean=0,std=1)
plt.figure()

# mean and variance for intial U,V, which determine Q
mean = [0.01, 0.1, 1, 10]
sigma = [0.01, 0.1, 1, 10]

Q_0 = []
for i in range(4):
    phi, S, psih = torch.linalg.svd(Y)
    psi = psih.transpose(-2, -1).conj()

    torch.manual_seed(12345)
    U_bar = torch.rand(m, k).normal_(mean=mean[i],std=sigma[i]) # 5 x 50
    torch.manual_seed(12345)
    V_bar = torch.rand(n, k).normal_(mean=mean[i],std=sigma[i]) # 10 x 50
    for ii in range(m):
        for j in range(k):
            if ii != j:
               U_bar[ii][j] = 0
    for ii in range(n):
        for j in range(k):
            if ii != j:
               V_bar[ii][j] = 0
    Q_0.append(torch.trace(torch.matmul(U_bar, U_bar.T)) + torch.trace(torch.matmul(V_bar, V_bar.T)))

    U = torch.matmul(phi, U_bar)
    V = torch.matmul(psi, V_bar)

    model = TwoLayerRadial(init_U=U, init_V=V)
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    loss_arr = []
    Q_arr = []
    for epoch in range(int(1e6)):
        loss, Q_train = train_epoch(epoch, X, Y)
        loss_arr.append((loss / torch.norm(Y)).detach().numpy())
        Q_arr.append(Q_train.detach().numpy())

    # plot loss
    plt.plot(loss_arr, label='Q={0:.3g}'.format(Q_0[i]), linewidth=2)


plt.yscale('log')
plt.xticks([2e5, 4e5, 6e5, 8e5, 10e5], fontsize=14)
plt.yticks([1e-10, 1e-7, 1e-4, 1e-1], fontsize=14)
plt.xlim(-0.5e5, 10e5 + 0.5e5)

plt.xlabel('Training steps', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(fontsize=13, loc='upper right', bbox_to_anchor=(1.0, 0.75))
plt.savefig('figures/Q-convergence/radial_zoomed_1e6.pdf'.format("radial"), bbox_inches='tight')
