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

NSIDE = 1
sigma_rbf = 100000
N_PROCESS_MAX = 45
N_sample = 150

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
COSMO_PARAMS_SIGMA = [0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]


def main(NSIDE):
    with open("B3DCMB/data/reference_data_right_beta_sync", "rb") as f:
        reference_data = pickle.load(f)

    sky_map = np.array(reference_data["sky_map"])

    sampler = Sampler(NSIDE)
    time_start = time.time()
    pool = mp.Pool(N_PROCESS_MAX)
    all_results = pool.map(sampler.sample_model, (sky_map for _ in range(N_sample)))
    time_elapsed = time.time() - time_start
    print(time_elapsed)

    with open("B3DCMB/data/simulated_sample_small", "wb") as f:
        pickle.dump(all_results, f)

    with open("B3DCMB/data/simulated_sample_small", "rb") as f:
        samples = pickle.load(f)

    log_weights = []
    for res in samples:
        log_weights.append(res["log_weight"])

    log_weights = np.array(log_weights)
    print(log_weights)
    print("\n")
    print(log_weights - np.max(log_weights))
    print(np.max(log_weights))
    print("\n")
    w = np.exp(log_weights - np.max(log_weights))
    print(w)
    print("\n")
    w = w/np.sum(w)
    print(w)

    ess = (np.sum(w)**2)/np.sum(w**2)
    print(ess)
    '''
    plt.hist(log_weights)
    plt.title("Log weights histogram")
    plt.savefig("B3DCMB/figures/log_weights_histogram.png")
    plt.close()

    plt.hist(w)
    plt.title("Histogram of weights")
    plt.savefig("B3DCMB/figures/weights_histogram.png")
    plt.close()
    '''

if __name__=='__main__':
    main(NSIDE)
