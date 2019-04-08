import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport *
from scipy.linalg.cython_lapack cimport *
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
def compute_exponent(double[:,:,:] sigmas_symm, double[:,::1] b, int l):

    cdef:
        int n = sigmas_symm.shape[1]
        int nrhs = 1
        int lda = sigmas_symm.shape[1]
        int ldb = sigmas_symm.shape[1]
        int info = 0
        int inc = 1
        double result = 0.0
        int[::1] pivot = np.zeros(sigmas_symm.shape[1], dtype = np.intc, order = "F")
        double[::1] current = np.zeros(sigmas_symm.shape[1], order = "F")
        double[::1, :] sigm_current = np.zeros((sigmas_symm.shape[1], sigmas_symm.shape[1]), order = "F")
        double out = 0.0

    for i in range(l):
        #current = b.base[i].copy_fortran()
        current = b[i].copy_fortran()
        sigm_current = sigmas_symm[i,:,:].copy_fortran()
        dgesv(&n, &nrhs, &sigm_current[0, 0], &lda, &pivot[0], &current[0], &ldb, &info)
        out = ddot(&n, &current[0], &inc, &b[i, 0], &inc)
        result = result + out

    return result



