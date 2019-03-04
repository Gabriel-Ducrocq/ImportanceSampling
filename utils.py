import numpy as np
import scipy
import healpy as hp
import matplotlib.pyplot as plt

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
COSMO_PARAMS_SIGMA = [0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]

def read_template(path, NSIDE, fields = (0, 1, 2, 3, 4, 5) ):
    map_ = hp.read_map(path, field=fields)
    map_ = hp.ud_grade(map_, nside_out=NSIDE)
    return map_

def compute_Sigma_Q_U(map4, map2, map3, map5):
    return map4 - map2, map3 - map5

def create_mean_var(path, NSIDE):
    map_ = read_template(path, NSIDE)
    Q, U = map_[0], map_[1]
    sigma_Q, sigma_U = compute_Sigma_Q_U(map_[4], map_[2], map_[3], map_[5])
    return Q.tolist(), U.tolist(), sigma_Q.tolist(), sigma_U.tolist()

def get_pixels_params(NSIDE):
    Q_sync, U_sync, sigma_Q_sync, sigma_U_sync = create_mean_var(
        'B3DCMB/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits', NSIDE)
    Q_dust, U_dust, sigma_Q_dust, sigma_U_dust = create_mean_var(
        'B3DCMB/COM_CompMap_DustPol-commander_1024_R2.00.fits', NSIDE)
    params = {"dust":{"mean":{"Q": Q_dust, "U": U_dust},
                    "sigma":{"Q": sigma_Q_dust, "U": sigma_U_dust}},
             "sync":{"mean":{"Q": Q_sync, "U": U_sync},
                    "sigma":{"Q": sigma_Q_sync, "U": sigma_U_sync}}}

    return params

def get_mixing_matrix_params(NSIDE):
    temp_dust, sigma_temp_dust, beta_dust, sigma_beta_dust = read_template(
        'B3DCMB/COM_CompMap_dust-commander_0256_R2.00.fits', NSIDE, fields =(3,5,6,8))
    beta_sync = hp.read_map('B3DCMB/sync_beta.fits', field=(0))
    beta_sync = hp.ud_grade(beta_sync, nside_out=NSIDE)
    sigma_beta_sync = sigma_beta_dust / (10 * np.std(sigma_beta_dust))
    params = {"dust":{"temp":{"mean": temp_dust.tolist(), "sigma":(sigma_temp_dust).tolist()},
                      "beta":{"mean":beta_dust.tolist(), "sigma": (sigma_beta_dust).tolist()}},
            "sync":{"beta":{"mean":beta_sync.tolist(), "sigma": (sigma_beta_sync).tolist()}}}

    return params

def aggregate_by_pixels_params(params):
    Q_sync = params["sync"]["mean"]["Q"]
    U_sync = params["sync"]["mean"]["U"]
    Q_dust = params["dust"]["mean"]["Q"]
    U_dust = params["dust"]["mean"]["U"]
    Qs = [[l[0], l[1]] for l in zip(Q_dust, Q_sync)]
    Us = [[l[0], l[1]] for l in zip(U_dust, U_sync)]

    sigma_Q_sync = params["sync"]["sigma"]["Q"]
    sigma_U_sync = params["sync"]["sigma"]["U"]
    sigma_Q_dust = params["dust"]["sigma"]["Q"]
    sigma_U_dust = params["dust"]["sigma"]["U"]
    sigma_Qs = [[l[0], l[1]] for l in zip(sigma_Q_dust, sigma_Q_sync)]
    sigma_Us = [[l[0], l[1]] for l in zip(sigma_U_dust, sigma_U_sync)]
    return Qs, Us, sigma_Qs, sigma_Us

def aggregate_pixels_params(params):
    Q_sync = params["sync"]["mean"]["Q"]
    U_sync = params["sync"]["mean"]["U"]
    Q_dust = params["dust"]["mean"]["Q"]
    U_dust = params["dust"]["mean"]["U"]
    templates_map = np.hstack([Q_sync, U_sync, Q_dust, U_dust])
    sigma_Q_sync = params["sync"]["sigma"]["Q"]
    sigma_U_sync = params["sync"]["sigma"]["U"]
    sigma_Q_dust = params["dust"]["sigma"]["Q"]
    sigma_U_dust = params["dust"]["sigma"]["U"]
    sigma_templates = [sigma_Q_sync, sigma_U_sync, sigma_Q_dust, sigma_U_dust]
    return templates_map, scipy.linalg.block_diag(*[np.diag(s_) for s_ in sigma_templates])

#### Refaire cette fonction pour matcher ce qu'il faut
def aggregate_mixing_params(params):
    temp_dust = params["dust"]["temp"]["mean"]
    beta_dust = params["dust"]["beta"]["mean"]
    beta_sync = params["sync"]["beta"]["mean"]
    mean = np.hstack([beta_dust, temp_dust, beta_sync])
    sigma_temp_dust = params["dust"]["temp"]["sigma"]
    sigma_beta_dust = params["dust"]["beta"]["sigma"]
    sigma_beta_sync = params["sync"]["beta"]["sigma"]
    sigma = np.hstack([sigma_beta_dust, sigma_temp_dust, np.abs(sigma_beta_sync)])
    return mean, sigma

def RBF_kernel(x, sigma = 1):
    return np.exp(-0.5*x/sigma)

def compute_discrepency_L2(tuple_input):
    ref_data, simulated_data = tuple_input
    return np.sum(np.abs(ref_data - simulated_data)**2)

def compute_discrepency_Inf(tuple_input):
    ref_data, simulated_data = tuple_input
    return np.max(ref_data - simulated_data)


def compute_acceptance_rates(discrepencies, epsilons, title, path):
    ratios = []
    for eps in epsilons:
        ratios.append(np.mean(np.random.binomial(1, RBF_kernel(np.array(discrepencies),eps))))

    plt.plot(epsilons, ratios)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def histogram_posterior(weights, cosmo_sample, reference_cosmo):
    for i, name in enumerate(COSMO_PARAMS_NAMES):
        print(i)
        e = []
        for set_cosmos in cosmo_sample:
            e.append(set_cosmos["cosmo_params"][i])

        print("Length of e:" + str(len(e)))
        prior = np.random.normal(COSMO_PARAMS_MEANS[i], COSMO_PARAMS_SIGMA[i], 10000)
        plt.hist(prior, density=True, alpha=0.5, label="Prior", bins = 100)
        plt.hist(e, density = True, alpha = 0.5, label = "Posterior", weights = weights, bins = 10)
        plt.legend(loc='upper right')
        plt.title('Histogram parameter: '+name)
        plt.axvline(reference_cosmo[i], color='k', linestyle='dashed', linewidth=1)
        plt.savefig("B3DCMB/figures/histogram_NSIDE_512" + name + ".png")
        plt.close()


def graph_dist_vs_theta(discrepencies, cosmo_params, reference_cosmo):
    for i, name in enumerate(COSMO_PARAMS_NAMES):
        cosmo = []
        for j in range(len(discrepencies)):
            cosmo.append(cosmo_params[j][i])

        plt.plot(cosmo, discrepencies, "o")
        plt.axvline(reference_cosmo[i], color='k', linestyle='dashed', linewidth=1)
        plt.title("Discrepency vs " + name)
        plt.savefig("B3DCMB/figures/discrepency_vs_" + name + ".png")
        plt.close()



def graph_dist_vs_dist_theta(discrepencies, cosmo_params, reference_cosmo, betas = None, reference_betas = None):
    params_distances = []
    if not betas:
        for j in range(len(discrepencies)):
            params_distances.append(np.sqrt(np.sum((reference_cosmo - cosmo_params[j])**2)))

        plt.plot(params_distances, discrepencies, "o")
        plt.title("Discrepency vs params distances")
        plt.savefig("B3DCMB/figures/discrepency_vs_params_distances.png")
        plt.close()

    else:
        for j in range(len(discrepencies)):
            d1 = np.sum((reference_cosmo - cosmo_params[j])**2)
            d2 = np.sum((reference_betas - betas[j])**2)
            params_distances.append(np.sqrt(d1+d2))

        plt.plot(params_distances, discrepencies, "o")
        plt.title("Discrepency vs all params distances")
        plt.savefig("B3DCMB/figures/discrepency_vs_all_params_distances.png")
        plt.close()



