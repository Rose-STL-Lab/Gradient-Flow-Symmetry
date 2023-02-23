import torch
from torch import nn
import os
import numpy as np
from model import TwoLayerRadial
import matplotlib.pyplot as plt
import pickle

retrain = True # set to False if loss data has been created in cache/train_radial

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

    Q = torch.trace(torch.matmul(U, U.T)) + torch.trace(torch.matmul(V, V.T))

    return loss, Q, Q_imbalance


mean = [0.01, 0.1, 1, 10]
sigma = [0.01, 0.1, 1, 10] #[1, 0.1, 0.01, 0.001]

if retrain == True:
    Q_0 = []
    for i in range(4):
        phi, S, psih = torch.linalg.svd(Y)
        psi = psih.transpose(-2, -1).conj()
        # S_expand = torch.zeros(5, 10)
        # S_expand[:5, :5] = torch.diag(S[:5])
        # print(Y - torch.matmul(torch.matmul(phi, S_expand), psi.T))
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
        # print(Q_0[-1])

        U = torch.matmul(phi, U_bar)
        V = torch.matmul(psi, V_bar)

        model = TwoLayerRadial(init_U=U, init_V=V)
        device = torch.device("cpu")
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

        loss_arr = []
        Q_arr = []
        Q_arr_imbalance = []
        for epoch in range(int(1e6)):
            loss, Q_train, Q_train_imbalance = train_epoch(epoch, X, Y)
            loss_arr.append((loss / torch.norm(Y)).detach().numpy())
            Q_arr.append(Q_train.detach().numpy())
            Q_arr_imbalance.append(Q_train_imbalance.detach().numpy())
            # if epoch % 1e6 == 0:
            #     print(epoch, Q_arr[-1])

        with open('cache/train_radial/loss_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(loss_arr, f)
        with open('cache/train_radial/Q_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(Q_arr, f)
        with open('cache/train_radial/Q_imbalance_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(Q_arr_imbalance, f)

    with open('cache/train_radial/Q0.pkl', 'wb') as f:
        pickle.dump(Q_0, f)


# load Q0
with open('cache/train_radial/Q0.pkl', 'rb') as f:
    Q_0 = pickle.load(f)

# plot loss
plt.figure()  
# np.set_printoptions(precision=20)
for i in range(4):
    with open('cache/train_radial/loss_{}.pkl'.format(i), 'rb') as f:
        loss_all = pickle.load(f)
    plt.plot(loss_all, label='Q={0:.3g}'.format(Q_0[i]), linewidth=2)
    # print(loss_all[0], loss_all[-1])
plt.yscale('log')
# plt.xscale('log')
# plt.xticks([1e0, 1e2, 1e4, 1e6], fontsize=14)
# plt.xticks([5e6, 1e7], fontsize=14)
# plt.xticks([1e6, 2e6, 3e6, 4e6, 5e6], fontsize=14)


# plt.xticks([5e5, 10e5, 15e5, 20e5], fontsize=14)
# plt.yticks([1e-10, 1e-7, 1e-4, 1e-1], fontsize=14)
# plt.xlim(-0.5e5, 2e6 + 0.5e5)

plt.xticks([2e5, 4e5, 6e5, 8e5, 10e5], fontsize=14)
plt.yticks([1e-10, 1e-7, 1e-4, 1e-1], fontsize=14)
plt.xlim(-0.5e5, 10e5 + 0.5e5)

plt.xlabel('Training steps', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(fontsize=13, loc='upper right', bbox_to_anchor=(1.0, 0.75))
plt.savefig('figures/Q-convergence/radial_zoomed_1e6.pdf'.format("radial"), bbox_inches='tight')

# plot Q
# for i in range(4):
#     plt.figure()  
#     with open('cache/train_radial/Q_{}.pkl'.format(i), 'rb') as f:
#         Q_arr = pickle.load(f)
#     plt.plot(Q_arr, label='Q_0={0:.3g}'.format(Q_0[i]))
#     plt.yscale('log')
#     plt.xscale('log')
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.xlabel('Training steps', fontsize=14)
#     plt.ylabel('Q', fontsize=14)
#     plt.legend()
#     plt.savefig('figures_rebuttal/radial_Q_mean{}_log_log.png'.format(mean[i]), dpi=400, bbox_inches='tight')


