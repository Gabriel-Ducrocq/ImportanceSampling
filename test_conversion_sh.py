import healpy as hp
from classy import Class
import numpy as np
import time
from healpy import sphtfunc


L_MAX_SCALARS = 5000
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
cls = self.cosmo.lensed_cl(L_MAX_SCALARS)
eb_tb = np.zeros(shape=cls["tt"].shape)
_, Q, U = hp.synfast((cls['tt'], cls['ee'], cls['bb'], cls['te'], eb_tb, eb_tb), nside=self.NSIDE, new=True)
end_generation = time.clock() - start
self.cosmo.struct_cleanup()
self.cosmo.empty()

start = time.clock()
res = sphtfunc.map2alm((Q,U))
print(res.shape)
end_back = time.clock() - start

print(end_generation)
print(end_back)
