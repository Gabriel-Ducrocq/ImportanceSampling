import healpy as hp
from classy import Class
import numpy as np
import time
import matplotlib.pyplot as plt

PLOT = False
PLOT_HISTOGRAM = True
BURNING = 3000
cosmo = Class()
NSIDE=512
L_MAX_SCALARS=1500
N_alm = L_MAX_SCALARS**2 + 2*L_MAX_SCALARS
Npix = 12 * NSIDE ** 2

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])

proposal_sigma= COSMO_PARAMS_SIGMA/6

N_iteration = 50000





