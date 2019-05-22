import healpy as hp
from classy import Class
import numpy as np

cosmo = Class()
NSIDE=512
L_MAX_SCALARS=1500
Npix = 12 * NSIDE ** 2

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'


def sample_cls(theta):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': L_MAX_SCALARS,
              'lensing': LENSING}

    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(L_MAX_SCALARS)
    eb_tb = np.zeros(shape=cls["tt"].shape)
    cosmo.struct_cleanup()
    cosmo.empty()
    return cls, eb_tb

def sample_skymap(theta):
    cls, eb_tb = sample_cls(theta)
