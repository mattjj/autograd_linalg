# distutils: extra_compile_args = -O2 -w
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from scipy.linalg.cython_blas cimport dsymv, ddot, dgemm
from scipy.linalg.cython_lapack cimport dtrtrs, dpotrf, dpotrs, dpotri

def solve_triangular(double[::1,:,:] L, double[::1,:,:] X, trans=False, lower=False):
    '''Just like scipy.linalg.solve_triangular, except broadcasts over leading dimensions'''
    pass
