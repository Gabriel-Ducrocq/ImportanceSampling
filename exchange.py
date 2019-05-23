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
COSMO_PARAMS_MEANS = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])

proposal_sigma= COSMO_PARAMS_SIGMA/4

N_iteration = 1000

def sample_cls(theta):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': L_MAX_SCALARS,
              'lensing': LENSING}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, list(theta))}
    print(d)
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(L_MAX_SCALARS)
    eb_tb = np.zeros(shape=cls["tt"].shape)
    cosmo.struct_cleanup()
    cosmo.empty()
    return cls["tt"], eb_tb

def sample_skymap(cls, eb_tb):
    skymap = hp.synalm(cls, new=True)
    return skymap


def propose_theta(old_theta):
    return old_theta + np.dot(np.diag(proposal_sigma), np.random.normal(0, 1, size=len(COSMO_PARAMS_MEANS)))


def compute_prior(theta):
    det = np.product(COSMO_PARAMS_SIGMA**2)
    num = np.exp(-(1/2)*np.sum(((theta - COSMO_PARAMS_MEANS)**2)/COSMO_PARAMS_SIGMA**2))
    return num/det

def compute_ratio_skymap(skymap, cls, cls_prime, eb_tb, auxiliary):
    S_inv = np.array([1/elt for i,elt in enumerate(cls) for _ in range(2*i+1)])
    S_prime_inv = np.array([1/elt for i,elt in enumerate(cls_prime) for _ in range(2*i+1)])

    return np.exp(-(1/2)*(np.sum(((skymap)**2)*(S_prime_inv - S_inv)) + np.sum(((auxiliary)**2)*(S_inv - S_prime_inv))))

def compute_MH_ratio(skymap, auxiliary, cls, cls_prime, theta, theta_prime, eb_tb):
    skymap_ratio = compute_ratio_skymap(skymap, cls, cls_prime, eb_tb, auxiliary)
    return (compute_prior(theta_prime)/compute_prior(theta))* skymap_ratio



TRUE_THETA = COSMO_PARAMS_MEANS - 0.5*COSMO_PARAMS_SIGMA
true_cls, eb_tb = sample_cls(TRUE_THETA)
print(true_cls[-50:])
TRUE_SKYMAP = sample_skymap(true_cls, eb_tb)

old_theta = COSMO_PARAMS_MEANS + np.diag(COSMO_PARAMS_SIGMA)*np.random.normal(0, 1, size = COSMO_PARAMS_MEANS.shape[0])
old_cls, old_eb_tb = sample_cls(old_theta)

for i in range(N_iteration):
    print(i)
    new_theta = propose_theta(old_theta)
    new_cls, eb_tb = sample_cls(new_theta)
    auxi_skymap = sample_skymap(new_cls, eb_tb)
    ratio = compute_MH_ratio(TRUE_SKYMAP, auxi_skymap, old_cls, new_cls, old_theta, new_theta, eb_tb)

    u = np.random.uniform()
    if u < ratio:
        old_theta = new_theta
        old_cls = new_cls




