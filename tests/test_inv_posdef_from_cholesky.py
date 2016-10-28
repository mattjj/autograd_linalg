from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd.numpy.linalg import solve
from itertools import product

from autograd_linalg.linalg import inv_posdef_from_cholesky
from autograd_linalg.util import T, symm

ndim = 5
leading_dims_options = [(), (3,), (3, 2)]

def rand_instance(leading_dims, ndim):
    dims = leading_dims + (ndim, ndim)
    square = lambda X: np.matmul(X, T(X))
    A = square(npr.randn(*dims)) + 10 * np.eye(ndim)
    L = np.linalg.cholesky(A)
    return A, L

def test_forward():
    npr.seed(0)
    def check_forward(A, L):
        ans1 = np.linalg.inv(A)
        ans2 = inv_posdef_from_cholesky(L)
        assert np.allclose(ans1, ans2)

    for leading_dims in leading_dims_options:
        A, L = rand_instance(leading_dims, ndim)
        yield check_forward, A, L

def test_grad():
    npr.seed(0)
    fun = lambda L: to_scalar(inv_posdef_from_cholesky(L))
    fun2 = lambda L: to_scalar(grad(fun)(L))

    for leading_dims in leading_dims_options:
        A, L = rand_instance(leading_dims, ndim)
        yield check_grads, fun, L
        yield check_grads, fun2, L
