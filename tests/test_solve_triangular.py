from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd.numpy.linalg import solve
from itertools import product

from autograd_linalg.linalg import solve_triangular
from autograd_linalg.util import T

ndim = 5

trans_options = ['T', 'N', 0, 1]
lower_options = [True, False]
leading_dims_options = [(), (3,), (3, 2)]
nrhs_options = [(), (2,)]
options = list(product(leading_dims_options, nrhs_options,
                       lower_options, trans_options))

def rand_instance(leading_dims, nrhs, ndim, lower):
    tri = np.tril if lower else np.triu
    L = tri(npr.normal(size=leading_dims + (ndim, ndim))) + 10. * np.eye(ndim)
    x = npr.normal(size=leading_dims + (ndim,) + nrhs)
    return L, x

def test_forward():
    npr.seed(0)
    def check_forward(L, x, trans, lower):
        ans1 = solve(T(L) if trans in (1, 'T') else L, x)
        ans2 = solve_triangular(L, x, lower=lower, trans=trans)
        assert np.allclose(ans1, ans2)

    for leading_dims, nrhs, lower, trans in options:
        L, x = rand_instance(leading_dims, nrhs, ndim, lower)
        yield check_forward, L, x, trans, lower

def test_grad_arg0():
    npr.seed(0)
    for leading_dims, nrhs, lower, trans in options:
        L, x = rand_instance(leading_dims, nrhs, ndim, lower)
        def fun(L):
            return to_scalar(solve_triangular(L, x, trans=trans, lower=lower))
        yield check_grads, fun, L

def test_grad_arg1():
    npr.seed(0)
    for leading_dims, nrhs, lower, trans in options:
        L, x = rand_instance(leading_dims, nrhs, ndim, lower)
        def fun(x):
            return to_scalar(solve_triangular(L, x, trans=trans, lower=lower))
        yield check_grads, fun, x
