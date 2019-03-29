import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport *
from scipy.linalg.cython_lapack cimport *
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
        int[:] pivot = np.zeros(sigmas_symm.shape[1], dtype = np.intc)
        double[:] current = np.zeros(sigmas_symm.shape[1], dtype = np.intc)
        double out = 0.0

    for i in range(l):
        current = b.base[i].copy()
        dgesv(&n, &nrhs, &sigmas_symm[i, 0, 0], &lda, &pivot[0], &current[0], &ldb, &info)
        out = ddot(&n, &current[0], &inc, &b[i, 0], &inc)
        result = result + out

    return result



