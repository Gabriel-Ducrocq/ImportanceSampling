import healpy as hp
from classy import Class
import numpy as np
import time
import matplotlib.pyplot as plt

BURNING = 3000
cosmo = Class()
NSIDE=512
L_MAX_SCALARS=50
Npix = 12 * NSIDE ** 2
cosmo = Class()

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])
COSMO_PARAMS_LOWER = np.array([0.7, 0.02, 0.01, 0.01, 2.5, 0.01])
COSMO_PARAMS_UPPER = np.array([1.4, 0.1, 1.2, 1.2, 3.5, 0.1])
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])

def noise_covariance_in_freq(nside):
    cov = LiteBIRD_sensitivities ** 2 / hp.nside2resol(nside, arcmin=True) ** 2
    return cov[0]/1000000

noise_covar_one_pix = noise_covariance_in_freq(NSIDE)

def proposal_theta():
    #return np.random.uniform(COSMO_PARAMS_LOWER, COSMO_PARAMS_UPPER)
    return COSMO_PARAMS_SIGMA*np.random.normal(0, 1, size = 6)+ COSMO_PARAMS_MEAN

def compute_cls(theta):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': L_MAX_SCALARS,
              'lensing': LENSING}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    print(cosmo.get_current_derived_parameters(["n_s"]))
    cls = cosmo.lensed_cl(L_MAX_SCALARS)
    eb_tb = np.zeros(shape=cls["tt"].shape)
    cosmo.struct_cleanup()
    cosmo.empty()
    all_cls = np.array([elt for i,elt in enumerate(cls["tt"]) for _ in range(2*i+1)])
    return all_cls

def sample_alm(cls):
    return np.sqrt(cls)*np.random.normal(0, 1, size = len(cls))

def sample_skymap(theta):
    cls = compute_cls(theta)
    return sample_alm(cls)

def compute_likelihood(skymap_alms):
    #skymap_pix = hp.sphtfunc.alm2map(skymap_alms.astype(complex), nside= NSIDE)
    skymap_pix = skymap_alms
    var = noise_covariance_in_freq(NSIDE)
    print(var)
    log_likelihood = -(1/2)*np.sum((((observed_skymap - skymap_pix)**2)/var)) - (1/2)*np.log(2*np.pi*var)*len(skymap_pix)
    return log_likelihood

TRUE_COSMO_PARAMS = COSMO_PARAMS_MEAN-5*COSMO_PARAMS_SIGMA
observed_alms = sample_skymap(TRUE_COSMO_PARAMS)
#observed_skymap = hp.sphtfunc.alm2map(observed_alms.astype(complex), nside = NSIDE)
observed_skymap =observed_alms
sampled_thetas = []
log_weights = []
print("Done observed skymap")
for i in range(100):
    print(i)
    new_theta = proposal_theta()
    new_skymap = sample_skymap(new_theta)
    log_weight = compute_likelihood(new_skymap)
    sampled_thetas.append(new_theta)
    log_weights.append(log_weight)

log_weights = np.array(log_weights)
w = np.exp(log_weights - np.max(log_weights))
normalized_weights = w/np.sum(w)
print(normalized_weights)
print(np.sum(normalized_weights))
ess = np.sum(w)**2/np.sum(w**2)
print(ess)
print(np.mean(np.array(sampled_thetas), axis = 0))
print(TRUE_COSMO_PARAMS)