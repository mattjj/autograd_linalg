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
options = list(product(leading_dims_options, lower_options, trans_options))

def test_forward():
    def check_forward(L, x, trans, lower):
        ans1 = solve(T(L) if trans in (1, 'T') else L, x)
        ans2 = solve_triangular(L, x, lower=lower, trans=trans)
        assert np.allclose(ans1, ans2)

    for dims, trans, lower in options:
        tri = np.tril if lower else np.triu
        L = tri(npr.normal(size=dims + (ndim, ndim)))
        x = npr.normal(size=dims + (ndim,))
        yield check_forward, L, x, trans, lower

# def grad_arg0():
#     for leading_dims in [(), 5, (5, 5)]:
#         for trans in ['T', 'N']:
#             L = np.tril(npr.normal(size=leading_dims + (N, N)))
#             x = npr.normal(size=leading_dims + (N,))
#             yield check_grads, 

# def grad_arg1():
#     pass
