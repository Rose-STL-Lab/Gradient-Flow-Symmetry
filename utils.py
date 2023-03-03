import torch
import torch.nn as nn
from torch.nn import Parameter

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()


def compute_Q(U, V, activation):
    """ Compute conserved quantity between two layers with elementwise activation and whitened input. 

    Args:
        U: An m-by-h matrix. 
        V: An n-by-h matrix. 
        activation: name of the activation function, one of ['linear', 'relu', 'tanh', 'sigmoid']

    Returns:
        A scalar $Q = \frac{1}{2} \Tr[U^TU] - \sum_{a,j} \int_{x_0}^{V_{aj}} dx \frac{\sigma(x)}{\sigma'(x)}$.
    """
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
    return Q


def loss_2norm(U, V, Y, activation):
    """ Compute L2 loss of a two-layer network with elementwise activation and whitened input.

    Args:
        U: An m-by-h matrix, parameters of the second layer.
        V: An n-by-h matrix, parameters of the first layer. 
        Y: An m-by-n matrix, the label.
        activation: name of the activation function, one of ['linear', 'relu', 'tanh', 'sigmoid']

    Returns:
        A scalar norm_Y_X = ||Y - U activation(V^T)||.
    """
    if activation == 'linear':
        norm_Y_X = torch.norm(Y - torch.matmul(U, V.T))
    elif activation == 'relu':
        norm_Y_X = torch.norm(Y - torch.matmul(U, relu(V.T)))
    elif activation == 'tanh':
        norm_Y_X = torch.norm(Y - torch.matmul(U, tanh(V.T)))
    elif activation == 'sigmoid':
        norm_Y_X = torch.norm(Y - torch.matmul(U, sigmoid(V.T)))
    else:
        raise 
    return norm_Y_X


def get_U_V(model):
    """ Get parameters from a two-layer model. 

    Args:
        model: an instance of TwoLayerElementwise (defined in model.py)

    Returns:
        U: An m-by-h matrix, parameters of the second layer.
        V: An n-by-h matrix, parameters of the first layer. 
    """
    param_list = []
    for param in model.parameters():
        param_list.append(param.data)
    U = param_list[0].clone() # 5 x 50
    V = param_list[1].clone() # 10 x 50
    return U, V