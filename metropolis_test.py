import healpy as hp
from classy import Class
import numpy as np


cosmo = Class()
NSIDE=512
L_MAX_SCALARS=1500
Npix = 12 * NSIDE ** 2

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
COSMO_PARAMS_SIGMA = [0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]

def proposal(old_theta):
    return old_theta + np.dot(np.diag(COSMO_PARAMS_SIGMA)/500, np.random.normal(0, 1, size = len(COSMO_PARAMS_MEANS)))

def sample_power_spectrum(cosmo_params):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': L_MAX_SCALARS,
              'lensing': LENSING}

    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, cosmo_params)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(L_MAX_SCALARS)
    eb_tb = np.zeros(shape=cls["tt"].shape)
    cosmo.struct_cleanup()
    cosmo.empty()
    print("SHAPE")
    print(cls["tt"].shape)
    return cls["tt"]

cls_tt = sample_power_spectrum([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
TRUE_MAP = hp.synalm(cls_tt, new=True)
print("TRUE MAP shape")
print(TRUE_MAP.shape)

def compute_posterior(cls, params):
    prior = (-1/2)*np.sum(((params - np.array(COSMO_PARAMS_MEANS))**2)/(np.array(COSMO_PARAMS_SIGMA)**2)) - (1/2)*(np.sum(np.log(np.array(COSMO_PARAMS_SIGMA)**2)))
    likelihood = (-1/2)*np.sum(TRUE_MAP/np.array([val for l, val in enumerate(cls) for _ in range(l+1)])) - (1/2)*np.sum(np.log(np.array([val for l, val in enumerate(cls) for m in range(l+1)])))
    return prior + likelihood

def kernel(old_theta):
    new_theta = proposal(old_theta)
    new_cls = sample_power_spectrum(new_theta)
    old_cls = sample_power_spectrum(old_theta)
    ratio = min(0, compute_posterior(new_cls, new_theta) - compute_posterior(old_cls, old_theta))
    u = np.random.uniform()
    if np.log(u) < ratio:
        return new_theta, 1

    return old_theta, 0


path = []
acceptance = []
init_theta = np.random.multivariate_normal(COSMO_PARAMS_MEANS, np.diag(COSMO_PARAMS_SIGMA))
path.append(init_theta)
current_theta = init_theta
for i in range(1000):
    if i%10 == 0:
        print(i)
        print(np.mean(acceptance))

    current_theta, accepted = kernel(current_theta)
    path.append(current_theta)
