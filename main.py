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

NSIDE = 512
sigma_rbf = 100000
N_PROCESS_MAX = 45
N_sample = 2

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
COSMO_PARAMS_SIGMA = [0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]

def main(NSIDE):
    sampler = Sampler(NSIDE)
    '''
    start_time = time.time()
    ref = sampler.sample_data()
    with open("B3DCMB/data/reference_data_As_NSIDE_512", "wb") as f:
        pickle.dump(ref, f)

    print(time.time() - start_time)

    #start = time.time()
    #data = sampler.sample_data()
    #print("Sampling true data in:")
    #print(time.time() - start)

    start = time.time()
    pool1 = mp.Pool(N_PROCESS_MAX)
    all_sigmas_squared = pool1.map(sampler.sample_data, (i for i in range(N_sample)))
    print(time.time() - start)

    with open("B3DCMB/data/all_sigmas", "wb") as f:
        pickle.dump(all_sigmas_squared, f)

    plt.hist(np.sqrt(np.array(all_sigmas_squared)), density=True, bins=50)
    plt.title('Histogram sigma')
    plt.savefig("B3DCMB/figures/histogram_sigmas.png")
    plt.close()
    print("Empirical means of sigmas:")
    print(np.mean(np.sqrt(np.array(all_sigmas_squared))))

    #with open("B3DCMB/data/reference_data_As_NSIDE_64", "wb") as f:
    #    pickle.dump(data, f)
    '''
    with open("B3DCMB/data/reference_data_As_NSIDE_512", "rb") as f:
        reference_data = pickle.load(f)

    print("Data opened")
    map = np.array(reference_data["sky_map"])
    config.sky_map = map

    time_start = time.time()
    pool1 = mp.Pool(N_PROCESS_MAX)
    pool2 = mp.Pool(N_PROCESS_MAX)
    noise_level = 0
    print("Starting sampling")
    all_sample = pool1.map(sampler.sample_model, (i for i in range(N_sample)))
    '''
    print("starting weight computing")
    log_weights = pool2.map(sampler.compute_weight, ((data, noise_level, i,) for i,data in enumerate(all_sample)))
    time_elapsed = time.time() - time_start
    print(time_elapsed)

    with open("B3DCMB/data/simulated_AS_NSIDE_512", "wb") as f:
        pickle.dump({"simulated_points":all_sample, "log_weights":log_weights},f)
    '''
    '''
    with open("B3DCMB/data/reference_data_As_NSIDE_512", "rb") as f:
        reference_data = pickle.load(f)


    with open("B3DCMB/data/simulated_AS_NSIDE_512", "rb") as f:
        all_results = pickle.load(f)

    log_weights = all_results["log_weights"]
    AS = []
    for params in all_results["simulated_points"]:
        AS.append(params["cosmo_params"][4])


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
    main(NSIDE)
