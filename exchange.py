import healpy as hp
from classy import Class
import numpy as np
import time
import matplotlib.pyplot as plt

PLOT = True

cosmo = Class()
NSIDE=512
L_MAX_SCALARS=1500
Npix = 12 * NSIDE ** 2

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])

proposal_sigma= COSMO_PARAMS_SIGMA/6

N_iteration = 5000

def sample_cls(theta):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': L_MAX_SCALARS,
              'lensing': LENSING}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, list(theta))}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(L_MAX_SCALARS)
    eb_tb = np.zeros(shape=cls["tt"].shape)
    cosmo.struct_cleanup()
    cosmo.empty()
    all_cls = np.array([elt for i,elt in enumerate(cls["tt"][2:]) for _ in range(2*(i+2)+1)])
    return all_cls, eb_tb

def sample_skymap(cls, eb_tb):
    skymap = np.sqrt(cls)*np.random.normal(0, 1, size = len(cls))
    return skymap


def propose_theta(old_theta):
    return old_theta + np.dot(np.diag(proposal_sigma), np.random.normal(0, 1, size=len(COSMO_PARAMS_MEANS)))


def compute_prior(theta):
    det = np.product(COSMO_PARAMS_SIGMA**2)
    num = np.exp(-(1/2)*np.sum(((theta - COSMO_PARAMS_MEANS)**2)/COSMO_PARAMS_SIGMA**2))
    return num/det

def compute_ratio_skymap(skymap, cls, cls_prime, eb_tb, auxiliary):
    S_inv = 1/cls
    S_prime_inv = 1/cls_prime

    return np.exp(-(1/2)*(np.sum(((skymap)**2)*(S_prime_inv - S_inv)) + np.sum(((auxiliary)**2)*(S_inv - S_prime_inv))))

def compute_MH_ratio(skymap, auxiliary, cls, cls_prime, theta, theta_prime, eb_tb):
    skymap_ratio = compute_ratio_skymap(skymap, cls, cls_prime, eb_tb, auxiliary)
    return (compute_prior(theta_prime)/compute_prior(theta))* skymap_ratio


"""
TRUE_THETA = COSMO_PARAMS_MEANS
true_cls, eb_tb = sample_cls(TRUE_THETA)
TRUE_SKYMAP = sample_skymap(true_cls, eb_tb)

old_theta = COSMO_PARAMS_MEANS + 4*COSMO_PARAMS_SIGMA*np.random.normal(0, 1, size = COSMO_PARAMS_MEANS.shape[0])
old_cls, old_eb_tb = sample_cls(old_theta)

path = []
accepted = 0
for i in range(N_iteration):
    print(i)
    stat_time = time.time()
    new_theta = propose_theta(old_theta)
    new_cls, eb_tb = sample_cls(new_theta)
    auxi_skymap = sample_skymap(new_cls, eb_tb)
    ratio = compute_MH_ratio(TRUE_SKYMAP, auxi_skymap, old_cls, new_cls, old_theta, new_theta, eb_tb)

    u = np.random.uniform()
    if u < ratio:
        old_theta = new_theta
        old_cls = new_cls
        print("accepted")
        accepted += 1

    print(ratio)
    path.append(old_theta)
    print(old_theta)
    print("\n")
    if i%100 == 0:
        print(i)
        print(accepted/(i+1))

print(accepted/N_iteration)
one = [l[0] for l in path]
plt.plot(one)
plt.savefig("B3DCMB/exchange.png")
np.save("B3DCMB/exhange.npy", np.array(path))

"""

if PLOT:
    path = np.load("B3DCMB/exhange.npy")
    first, second, third, fourth, fifth, sixth = zip(*path)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.7, hspace=0.7)
    fig.suptitle("Trajectories")
    axes[0,0].plot(first)
    axes[0,0].set_title("n_s")

    axes[0,1].plot(second)
    axes[0,1].set_title("omega_b")

    axes[1,0].plot(third)
    axes[1,0].set_title("omega_cdm")

    axes[1,1].plot(fourth)
    axes[1,1].set_title("100*theta_s")

    axes[2,0].plot(fifth)
    axes[2,0].set_title("ln10^{10}A_s")

    axes[2,1].plot(sixth)
    axes[2,1].set_title("tau_reio")

    fig.savefig("B3DCMB/exchange.png")

