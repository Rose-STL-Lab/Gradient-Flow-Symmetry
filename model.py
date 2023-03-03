import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import Parameter

class TwoLayerRadial(nn.Module):
    def __init__(self, init_U, init_V):
        super(TwoLayerRadial, self).__init__()
        self.U = Parameter(init_U.clone())
        self.V = Parameter(init_V.clone())

    def forward(self, x):
        V_T = self.V.T
        V_T_square = torch.square(V_T)
        row_norm_square = torch.sum(V_T_square, dim=1)
        row_norm_square = row_norm_square - 0
        g_V_T = V_T / row_norm_square[:,None]

        out = torch.matmul(self.U, g_V_T)
        return out

class TwoLayerElementwise(nn.Module):
    def __init__(self, init_U, init_V, activation):
        super(TwoLayerElementwise, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.U = Parameter(init_U.clone())
        self.V = Parameter(init_V.clone())
        self.activation = activation

    def forward(self, x):
        out = torch.matmul(self.V.T, x)

        if self.activation == 'relu':
            out = self.relu(out)
        elif self.activation == 'tanh':
            out = self.tanh(out)
        elif self.activation == 'sigmoid':
            out = self.sigmoid(out)

        out = torch.matmul(self.U, out)
        return out