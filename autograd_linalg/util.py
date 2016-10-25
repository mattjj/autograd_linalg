import autograd.numpy as np

T = lambda x: np.swapaxes(x, -1, -2)
symm = lambda x: (x + T(x)) / 2.
al2d = lambda x: x if np.ndim(x) > 1 else x[...,None]  # assumes x is at least 1d
