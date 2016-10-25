# distutils: extra_compile_args = -O2 -w
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from scipy.linalg.cython_lapack cimport dtrtrs

def solve_triangular(double[::1,:,:] L, double[::1,:,:] X, trans, lower):
    cdef:
        int i, K = L.shape[2]
        double[::1,:,:] out = np.empty_like(X, dtype=np.double, order='F')
        char _trans = 'T' if trans in (1, 'T') else 'N'
        char _lower = 'L' if lower else 'U'

    for i in range(K):
        _solve_triangular(L[:,:,i], X[:,:,i], out[:,:,i], _trans, _lower)

    return np.asarray(out)

cdef inline void _solve_triangular(
        double[::1,:] L, double[::1,:] X, double[::1,:] out,
        char trans, char lower):
    cdef:
        int M = X.shape[0], N = X.shape[1], info = 0, i, j

    for j in range(N):
        for i in range(M):
            out[i,j] = X[i,j]
    dtrtrs(&lower, &trans, 'N', &M, &N, &L[0,0], &M, &out[0,0], &M, &info)
