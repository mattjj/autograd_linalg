import numpy as np
import autograd.numpy as anp
from autograd.core import primitive
from autograd.scipy.linalg import _flip

import cython_linalg as cyla
from util import T, symm, al2d

@primitive
def solve_triangular(a, b, trans=0, lower=False, **kwargs):
    flat_a = np.require(np.reshape(a, (-1,) + a.shape[-2:]), order='F')
    flat_b = np.require(np.reshape(b, flat_a.shape[:-1] + (1,)), order='F')
    flat_result = cyla.solve_triangular(flat_a, flat_b, trans=trans, lower=lower, **kwargs)
    return np.reshape(flat_result, b.shape)

def make_grad_solve_triangular(ans, a, b, trans=0, lower=False, **kwargs):
    tri = anp.tril if (lower ^ (_flip(a, trans) == 'N')) else anp.triu
    transpose = lambda x: x if _flip(a, trans) != 'N' else T(x)

    def solve_triangular_grad(g):
        v = al2d(solve_triangular(a, g, trans=_flip(a, trans), lower=lower))
        return -transpose(tri(anp.dot(v, T(al2d(ans)))))

    return solve_triangular_grad
solve_triangular.defgrad(make_grad_solve_triangular)

solve_trans = lambda L, X: solve_triangular(L, X, lower=True, trans='T')
conjugate_solve = lambda L, X: solve_trans(L, T(solve_trans(L, T(X))))
phi = lambda X: anp.tril(X) / 1. + anp.eye(X.shape[-1])
anp.linalg.cholesky.defgrad(lambda g: symm(conjugate_solve(L, phi(anp.matmul(L, g)))))
