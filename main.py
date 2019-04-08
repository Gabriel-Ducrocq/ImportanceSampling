import numpy as np
from sampler import Sampler
from utils import RBF_kernel, compute_discrepency_L2, compute_discrepency_Inf, compute_acceptance_rates, \
    histogram_posterior, graph_dist_vs_theta, graph_dist_vs_dist_theta
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
from scipy import stats
from matplotlib import cm
import config
import pandas as pd
from log_weight_computing import compute_exponent


from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import get_instrument
from utils import get_pixels_params, get_mixing_matrix_params, aggregate_pixels_params, aggregate_mixing_params, aggregate_by_pixels_params
from fgbuster.component_model import CMB, Dust, Synchrotron
import pysm
import healpy as hp
import scipy
import sys

NSIDE = 512
sigma_rbf = 100000
N_sample = 50
Npix = 12*NSIDE**2

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
COSMO_PARAMS_SIGMA = [0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])

Qs, Us, sigma_Qs, sigma_Us = aggregate_by_pixels_params(get_pixels_params(NSIDE))

instrument = pysm.Instrument(get_instrument('litebird', NSIDE))
components = [CMB(), Dust(150.), Synchrotron(150.)]
mixing_matrix = MixingMatrix(*components)
mixing_matrix_evaluator = mixing_matrix.evaluator(instrument.Frequencies)


def noise_covariance_in_freq(nside):
    cov = LiteBIRD_sensitivities ** 2 / hp.nside2resol(nside, arcmin=True) ** 2
    return cov

noise_covar_one_pix = noise_covariance_in_freq(NSIDE)


def sample_mixing_matrix_parallel(betas):
    return mixing_matrix_evaluator(betas)[:, 1:]


def prepare_sigma(input):
    sampled_beta, i = input
    mixing_mat = list(sample_mixing_matrix_parallel(sampled_beta))
    mean = np.dot(mixing_mat, config.arr_means[i])
    sigma = np.diag(noise_covar_one_pix) + np.einsum("ij,jk,lk", mixing_mat,
                                                          (np.diag(config.arr_sigmas[i]) ** 2),
                                                          mixing_mat)

    sigma_symm = np.asfortranarray((sigma + sigma.T) / 2)
    log_det = np.log(scipy.linalg.det(2 * np.pi * sigma_symm))
    return mean, sigma_symm, log_det








def main(NSIDE, run_num):
    start_time = time.time()
    sampler = Sampler(NSIDE)

    with open("B3DCMB/data/reference_data_free_beta_NSIDE_512", "rb") as f:
        reference_data = pickle.load(f)

    print("Data opened")
    map = np.array(reference_data["sky_map"])
    config.sky_map = map

    config.arr_sigmas = sigma_Qs + sigma_Us
    config.arr_means = Qs + Us
    print("Creating mixing matrix")

    _, sampled_beta = sampler.sample_model_parameters()
    sampled_beta = np.tile(sampled_beta, (2, 1))
    pool1 = mp.Pool(config.N_PROCESS_MAX)
    print("Launching")
    all_sample = pool1.map(prepare_sigma, ((sampled_beta[i, :], i,)
                                           for i in range(len(sampled_beta))), chunksize=25000)

    print("Unzipping result")
    means, sigmas_symm, log_det = zip(*all_sample)
    config.sigmas_symm = np.array(list(sigmas_symm))
    config.means = [i for l in means for i in l]
    config.denom = -(1 / 2) * np.sum(log_det)

    pool1 = mp.Pool(config.N_PROCESS_MAX)
    noise_level = 0
    print("Starting sampling")
    all_sample = pool1.map(sampler.sample_model, (i for i in range(N_sample)))

    config.N_PROCESS_MAX = 10
    print("starting weight computing")
    pool2 = mp.Pool(config.N_PROCESS_MAX)
    log_weights = pool2.map(sampler.compute_weight, ((noise_level, i)
                                                    for i in range(len(all_sample))))

    #with open("B3DCMB/data/simulated_beta_NSIDE_512_" + str(run_num), "wb") as f:
    #    pickle.dump({"simulated_points":all_sample, "sampled_beta":sampled_beta, "log_weights":log_weights},f)


    config.N_PROCESS_MAX = 50

    time_elapsed = time.time() - start_time
    print("Script number " + str(run_num) + " took " + str(time_elapsed) + "seconds")


    w = np.exp(log_weights - np.max(log_weights))
    w = w/np.sum(w)

    ess = (np.sum(w)**2)/np.sum(w**2)
    print(ess)
    print(time_elapsed)

    #with open("B3DCMB/data/reference_data_As_NSIDE_512", "rb") as f:
    #    reference_data = pickle.load(f)
    '''
    with open("B3DCMB/data/simulated_AS_NSIDE_512_reference", "rb") as f:
        ref = pickle.load(f)

    with open("B3DCMB/data/simulated_AS_NSIDE_512_sup", "rb") as f:
        sup = pickle.load(f)


    #log_weights = all_results["log_weights"]

    ref_log_weights = ref["log_weights"]
    sup_log_weights = sup["log_weights"]

    print(ref_log_weights)
    print(sup_log_weights)

    plt.hist(ref_log_weights, density=True, alpha=0.5, label="reference A_s", bins=10)
    plt.hist(sup_log_weights, density=True, alpha=0.5, label="A_s = 15", bins=10)
    plt.legend(loc='upper right')
    plt.title('Histogram log weights')
    #plt.axvline(reference_cosmo[i], color='k', linestyle='dashed', linewidth=1)
    plt.savefig("B3DCMB/figures/log_weights_histo.png")
    plt.close()
    '''

    '''
    log_weights = np.array(log_weights)
    print(log_weights)
    print("\n")
    print(log_weights - np.max(log_weights))
    print("\n")
    w = np.exp(log_weights - np.max(log_weights))
    print(np.sort(log_weights - np.max(log_weights))[-2:])
    print(w)
    print("\n")
    w = w/np.sum(w)
    print(w)

    ess = (np.sum(w)**2)/np.sum(w**2)
    print(ess)

    print([k for k in reference_data.keys()])
    histogram_posterior(w, all_results["simulated_points"], reference_data["cosmo_params"])
    '''
    '''
    plt.hist(log_weights, bins = 200)
    plt.title("Log weights histogram")
    plt.savefig("B3DCMB/figures/log_weights_histogram.png")
    plt.close()

    plt.boxplot(log_weights)
    plt.title("Log weights boxplot")
    plt.savefig("B3DCMB/figures/log_weights_boxplot.png")
    plt.close()

    plt.hist(w, bins = 200)
    plt.title("Histogram of weights")
    plt.savefig("B3DCMB/figures/weights_histogram.png")
    plt.close()

    e = np.sort(log_weights)
    print(e[-1])
    print(e[-2])
    print(e[-2] - e[-1])

    a = np.exp(log_weights)
    a /= np.sum(a)
    print(np.sum(a))
    print(np.sort(a)[::-1])
    '''



if __name__=='__main__':
    run_num = sys.argv[1]
    main(NSIDE, run_num)
