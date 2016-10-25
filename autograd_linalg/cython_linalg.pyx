# distutils: extra_compile_args = -O2 -w
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython cimport floating
from scipy.linalg.cython_lapack cimport dtrtrs, strtrs

def solve_triangular(floating[:,:,::1] L, floating[:,:,::1] X, trans, lower):
    cdef int i, K = L.shape[0]
    cdef floating[:,:,::1] out = np.empty_like(X)

    # flip these because we're working in C order
    cdef char _trans = 'T' if trans not in (1, 'T') else 'N'
    cdef char _lower = 'U' if lower else 'L'

    for i in range(K):
        _solve_triangular(L[i,:,:], X[i,:,:], out[i,:,:], _trans, _lower)

    return np.asarray(out)

cdef inline void _solve_triangular(
        floating[:,::1] L, floating[:,::1] X, floating[:,::1] out,
        char trans, char lower):
    cdef int M = X.shape[0], N = X.shape[1], info = 0, i, j

    for i in range(M):
        for j in range(N):
            out[i,j] = X[j,i]  # transpose because we're working in C order

    if floating is double:
        dtrtrs(&lower, &trans, 'N', &M, &N, &L[0,0], &M, &out[0,0], &M, &info)
    else:
        strtrs(&lower, &trans, 'N', &M, &N, &L[0,0], &M, &out[0,0], &M, &info)
