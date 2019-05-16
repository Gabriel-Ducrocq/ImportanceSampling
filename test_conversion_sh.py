import healpy as hp
from classy import Class
import numpy as np
import time
from healpy import sphtfunc


NSIDE = 512
L_MAX_SCALARS = 2000
LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
cosmo = Class()
start = time.clock()
cosmo_params = {l[0]: l[1] for l in zip(COSMO_PARAMS_NAMES, COSMO_PARAMS_MEANS)}
params = {'output': OUTPUT_CLASS,
          'l_max_scalars': L_MAX_SCALARS,
          'lensing': LENSING}
params.update(cosmo_params)
cosmo.set(params)
cosmo.compute()
cls = cosmo.lensed_cl(L_MAX_SCALARS)
eb_tb = np.zeros(shape=cls["tt"].shape)
I, Q, U = hp.synfast((cls['tt'], cls['ee'], cls['bb'], cls['te'], eb_tb, eb_tb), nside=NSIDE, new=True)
print(cls["tt"].shape)
end_generation = time.clock() - start
cosmo.struct_cleanup()
cosmo.empty()

start = time.clock()
res = sphtfunc.map2alm((I, Q, U), lmax = L_MAX_SCALARS)
#res = sphtfunc.map2alm(U)
print(res.shape)
print(cls["tt"].shape)
end_back = time.clock() - start

s = 0
for l in range(0, 20):
    s += 2*l+1

print(end_generation)
print(end_back)
print(s)

U = hp.synalm(cls['tt'], new=True)
print(U.shape)
