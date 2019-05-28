import healpy as hp
from classy import Class
import numpy as np
import time
import matplotlib.pyplot as plt

BURNING = 3000
cosmo = Class()
NSIDE=512
L_MAX_SCALARS=1500
Npix = 12 * NSIDE ** 2
cosmo = Class()

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
TRUE_COSMO_PARAMS = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])
COSMO_PARAMS_LOWER = np.array([0.7, 0.01, 0.01, 0.01, 2.5, 0.01])
COSMO_PARAMS_UPPER = np.array([1.4, 0.1, 1.2, 1.2, 3.5, 0.1])
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])

def noise_covariance_in_freq(nside):
    cov = LiteBIRD_sensitivities ** 2 / hp.nside2resol(nside, arcmin=True) ** 2
    return cov[0]

noise_covar_one_pix = noise_covariance_in_freq(NSIDE)

def proposal_theta():
    return np.random.uniform(COSMO_PARAMS_LOWER, COSMO_PARAMS_UPPER)

def compute_cls(theta):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': L_MAX_SCALARS,
              'lensing': LENSING}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    print(d)
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(L_MAX_SCALARS)
    eb_tb = np.zeros(shape=cls["tt"].shape)
    cosmo.struct_cleanup()
    cosmo.empty()
    all_cls = np.array([elt for i,elt in enumerate(cls["tt"][2:]) for _ in range(2*(i+2)+1)])
    return all_cls

def sample_alm(cls):
    return np.sqrt(cls)*np.random.normal(0, 1, size = len(cls))

def sample_skymap(theta):
    cls = compute_cls(theta)
    return sample_alm(cls)

def compute_likelihood(skymap_alm):
    skymap_pix = hp.spht.alm2map(skymap_alm, NSIDE, lmax = L_MAX_SCALARS)
    var = noise_covariance_in_freq(NSIDE)
    likelihood = np.exp(-(1/2)*((observed_skymap - skymap)**2)/var)/np.sqrt((2*np.pi*var)**len(skymap_pix))
    return likelihood

observed_skymap = sample_skymap(compute_cls(TRUE_COSMO_PARAMS))
sampled_thetas = []
weights = []
print("Done observed skymap")
for i in range(100):
    print(i)
    new_theta = proposal_theta()
    new_skymap = sample_skymap(new_theta)
    weight = compute_likelihood(new_skymap)
    sampled_thetas.append(new_theta)
    weights.append(weight)

weights = np.array(weights)
ess = np.sum(weigths)**2 / np.sum(weights**2)
print(ess)