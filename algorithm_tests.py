import numpy as np

from non_linear_action_algorithm import (Rmatrix, rectangular_to_spherical,
                                         spherical_to_rectangular)

"""Test the coordinate converstions"""
for _ in range(5):
    n = 1+int(10*np.random.random())
    z = np.random.random(n)
    zs = rectangular_to_spherical(z)
    zp = spherical_to_rectangular(zs)
    zps = rectangular_to_spherical(zp)
    assert np.max(zp - z) < 1e-9, "error"
    assert max([zps[i] - zs[i] for i in range(n)]) < 1e-9, "error"

"""Test that the R-matrix is a scalar matrix times an orthogonal matrix"""
"""Test that the first column of R is the original vector"""
for _ in range(5):
    n = 1+ int(10*np.random.random())
    z = np.random.random(n)
    r = np.linalg.norm(z)
    zs = rectangular_to_spherical(z)
    R = Rmatrix(zs)
    assert np.max( R @ np.transpose(R) - (r**2) * np.eye(n) ) < 1e-9, "error"
    assert np.max( R[:][0] - z ) <1e-9, "error"