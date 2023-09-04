from math import cos, pi, sin
from typing import List

import numpy as np

def rectangular_to_spherical(z: np.array) -> List:
    """Convert from rectangular coordinates to spherical coordinates"""
    r = np.linalg.norm(z)
    assert r != 0, "Input must be a nonzero vector"
    alpha = [r]
    product = r
    for i in range(len(z)-1):
        alphai = np.arccos(z[i]/ product)
        alpha.append(alphai)
        product *= np.sin(alphai)
    return alpha

def spherical_to_rectangular(coords: List) -> np.array:
    """Convert from spherical coordinates to rectangular coordinates"""
    assert coords[0] > 0, "Input must be a nonzero vector"
    z = []
    product = coords[0]
    for a in coords[1:]:
        z.append(cos(a)*product)
        product *= sin(a)
    z.append(product)
    assert len(z) == len(coords)
    return np.array(z)

def Rmatrix(coords: List) -> np.array:
    """The R-matrix for the non-linear action"""
    n = len(coords)
    r = coords[0]
    R = np.zeros((n,n))
    alphas= [0] + coords[1:] + [0]
    for j in range(n):
        prodcut = r
        for i in range(j,n):
            R[i][j] = cos(alphas[j])*prodcut*cos(alphas[i+1])
            prodcut *= sin(alphas[i+1])
    for i in range(n-1):
        R[i][i+1] = -r*sin(alphas[i+1])
    return np.array(R)







