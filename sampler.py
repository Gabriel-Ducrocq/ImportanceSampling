import numpy as np
import scipy
from scipy import stats
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import get_instrument
import healpy as hp
from classy import Class
import pysm
from utils import get_pixels_params, get_mixing_matrix_params, aggregate_pixels_params, aggregate_mixing_params, aggregate_by_pixels_params
from fgbuster.component_model import CMB, Dust, Synchrotron
import config
import matplotlib.pyplot as plt

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
MEAN_AS = 3.047
SIGMA_AS = 0.014
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
COSMO_PARAMS_SIGMA = [0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])
Nfreq = 15
L_MAX_SCALARS = 5000
LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'



class Sampler:
    def __init__(self, NSIDE):
        self.NSIDE = NSIDE
        self.Npix = 12*NSIDE**2
        print("Initialising sampler")
        self.cosmo = Class()
        print("Maps")
        self.Qs, self.Us, self.sigma_Qs, self.sigma_Us = aggregate_by_pixels_params(get_pixels_params(self.NSIDE))
        print("betas")
        self.matrix_mean, self.matrix_var = aggregate_mixing_params(get_mixing_matrix_params(self.NSIDE))
        print("Cosmo params")
        self.cosmo_means = np.array(COSMO_PARAMS_MEANS)
        self.cosmo_stdd = np.diag(COSMO_PARAMS_SIGMA)

        self.instrument = pysm.Instrument(get_instrument('litebird', self.NSIDE))
        self.components = [CMB(), Dust(150.), Synchrotron(150.)]
        self.mixing_matrix = MixingMatrix(*self.components)
        self.mixing_matrix_evaluator = self.mixing_matrix.evaluator(self.instrument.Frequencies)

        self.noise_covar_one_pix = self.noise_covariance_in_freq(self.NSIDE)
        self.noise_stdd_all = np.concatenate([np.sqrt(self.noise_covar_one_pix) for _ in range(2*self.Npix)])
        print("End of initialisation")

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        del state_dict["mixing_matrix_evaluator"]
        del state_dict["cosmo"]
        del state_dict["mixing_matrix"]
        del state_dict["components"]
        return state_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cosmo = Class()
        self.components = [CMB(), Dust(150.), Synchrotron(150.)]
        self.mixing_matrix = MixingMatrix(*self.components)
        self.mixing_matrix_evaluator = self.mixing_matrix.evaluator(self.instrument.Frequencies)

    def sample_normal(self, mu, stdd):
        standard_normal = np.random.normal(0, 1, size = mu.shape[0])
        standard_normal += mu
        normal = np.dot(stdd, standard_normal)
        return normal

    def noise_covariance_in_freq(self, nside):
        cov = LiteBIRD_sensitivities ** 2 / hp.nside2resol(nside, arcmin=True) ** 2
        return cov

    def sample_model_parameters(self):
        #sampled_cosmo = self.sample_normal(self.cosmo_means, self.cosmo_stdd)
        sampled_cosmo = np.array([0.9665, 0.02242, 0.11933, 1.04101, np.random.normal(MEAN_AS, SIGMA_AS), 0.0561])
        #sampled_beta = self.sample_normal(self.matrix_mean, np.diag(self.matrix_var)).reshape((self.Npix, -1), order = "F")
        sampled_beta = self.matrix_mean.reshape((self.Npix, -1), order = "F")
        return sampled_cosmo, sampled_beta

    def sample_CMB_QU(self, cosmo_params):
        params = {'output': OUTPUT_CLASS,
                  'l_max_scalars': L_MAX_SCALARS,
                  'lensing': LENSING}
        params.update(cosmo_params)
        self.cosmo.set(params)
        self.cosmo.compute()
        cls = self.cosmo.lensed_cl(L_MAX_SCALARS)
        eb_tb = np.zeros(shape=cls["tt"].shape)
        _, Q, U = hp.synfast((cls['tt'], cls['ee'], cls['bb'], cls['te'], eb_tb, eb_tb), nside=self.NSIDE, new=True)
        self.cosmo.struct_cleanup()
        self.cosmo.empty()
        return Q, U

    def sample_mixing_matrix(self, betas):
        mat_pixels = []
        for i in range(self.Npix):
            m = self.mixing_matrix_evaluator(betas[i,:])[:, 1:]
            mat_pixels.append(m)

        return mat_pixels

    def sample_mixing_matrix_full(self, betas):
        mat_pixels = []
        for i in range(self.Npix):
            m = self.mixing_matrix_evaluator(betas[i,:])
            mat_pixels.append(m)

        return mat_pixels

    def sample_model(self, input_params):
        print("Getting input param")
        random_seed = input_params
        print("Setting random seed")
        np.random.seed(random_seed)
        print("creating params")
        cosmo_params, sampled_beta = self.sample_model_parameters()
        print("Setting cosmo dict")
        cosmo_dict = {l[0]: l[1] for l in zip(COSMO_PARAMS_NAMES, cosmo_params.tolist())}
        print("Sampling cmb")
        tuple_QU = self.sample_CMB_QU(cosmo_dict)
        print("Concatenating")
        map_CMB = np.concatenate(tuple_QU)
        return {"map_CMB": map_CMB,"cosmo_params": cosmo_params,"betas": sampled_beta}

    def compute_weight(self, input):
        observed_data = config.sky_map
        data, noise_level, random_seed = input
        np.random.seed(random_seed)
        map_CMB = data["map_CMB"]
        sampled_beta = data["betas"]
        mixing_matrix = self.sample_mixing_matrix(sampled_beta)
        all_mixing_matrix = 2*mixing_matrix
        noise_addition = np.diag(noise_level*np.ones(Nfreq))
        means_and_sigmas = [[np.dot(l[0], l[1]), noise_addition + np.diag(
            self.noise_covar_one_pix) + np.einsum("ij,jk,lk", l[0], np.diag(l[2]), l[0])]
            for l in zip(all_mixing_matrix, self.Qs + self.Us, self.sigma_Qs + self.sigma_Us)]
        means, sigmas = zip(*means_and_sigmas)
        sigmas = [(s+s.T)/2 for s in sigmas]
        mean = np.array([i for l in means for i in l])
        duplicate_CMB = np.array([l for l in map_CMB for _ in range(15)])
        x = np.split((observed_data - duplicate_CMB) - mean, 24)
        log_det = np.sum([np.log(scipy.linalg.det(2*np.pi*s)) for s in sigmas])
        denom = -(1 / 2) * log_det
        lw = -(1/2)*np.sum([np.dot(l[1], scipy.linalg.solve(l[0], l[1].T)) for l in zip(sigmas, x)]) + denom
        return lw

    def sample_data(self):
        print("Sampling parameters")
        cosmo_params, sampled_beta = self.sample_model_parameters()
        print("Computing mean and cov of map")
        mean_map = np.array([i for l in self.Qs + self.Us for i in l])
        covar_map =[i for l in self.sigma_Qs + self.sigma_Us for i in l]
        print("Sampling maps Dust and Sync")
        maps = self.sample_normal(mean_map, np.diag(covar_map))
        print("Computing cosmo params")
        cosmo_dict = {l[0]: l[1] for l in zip(COSMO_PARAMS_NAMES, cosmo_params.tolist())}
        print("Sampling CMB signal")
        tuple_QU = self.sample_CMB_QU(cosmo_dict)
        map_CMB = np.concatenate(tuple_QU)
        print("Creating mixing matrix")
        mixing_matrix = self.sample_mixing_matrix(sampled_beta)
        print("Scaling to frequency maps")
        freq_maps = np.dot(scipy.linalg.block_diag(*2*mixing_matrix), maps.T)
        print("Adding CMB to frequency maps")
        duplicated_cmb = np.array([l for l in map_CMB for _ in range(15)])
        print("Creating noise")
        noise = self.sample_normal(np.zeros(2 * 15 * self.Npix),np.diag(self.noise_stdd_all))
        print("Adding noise to the maps")
        sky_map = freq_maps + duplicated_cmb + noise
        return {"sky_map": sky_map, "cosmo_params": cosmo_params, "betas": sampled_beta}
