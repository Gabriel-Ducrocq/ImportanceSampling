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
import matplotlib.pyplot as plt
import config
from itertools import chain, tee
import pickle
import time
import multiprocessing as mp

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

        print(self.Qs)

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


    def prepare_sigma(self, input):
        sampled_beta, i = input
        mixing_mat = self.sample_mixing_matrix_parallel(sampled_beta)
        mean = np.dot(mixing_mat, (self.Qs + self.Us)[2*i:(2*i+2)])
        print(mean.shape)
        sigma = np.diag(self.noise_covar_one_pix) + np.einsum("ij,jk,lk", mixing_mat,
                                                            (np.diag((self.sigma_Qs +self.sigma_Us)[i])**2), mixing_mat)

        sigma_symm = (sigma + sigma.T) / 2
        log_det = np.log(scipy.linalg.det(2 * np.pi * sigma_symm))
        return mean, sigma_symm, log_det

    def sample_mixing_matrix_parallel(self, betas):
        return self.mixing_matrix_evaluator(betas)[:, 1:]



    def sample_normal(self, mu, stdd, diag = False):
        standard_normal = np.random.normal(0, 1, size = mu.shape[0])
        if diag:
            normal = np.multiply(stdd, standard_normal)
        else:
            normal = np.dot(stdd, standard_normal)

        normal += np.add(mu, normal)
        return normal

    def noise_covariance_in_freq(self, nside):
        cov = LiteBIRD_sensitivities ** 2 / hp.nside2resol(nside, arcmin=True) ** 2
        return cov

    def sample_model_parameters(self):
        #sampled_cosmo = self.sample_normal(self.cosmo_means, self.cosmo_stdd)
        sampled_cosmo = np.array([0.9665, 0.02242, 0.11933, 1.04101, np.random.normal(MEAN_AS, 100*SIGMA_AS), 0.0561])
        #sampled_beta = self.sample_normal(self.matrix_mean, np.diag(self.matrix_var)).reshape((self.Npix, -1), order = "F")
        sampled_beta = self.matrix_mean.reshape((self.Npix, -1), order = "F")
        return sampled_cosmo, sampled_beta

    def sample_CMB_QU(self, cosmo_params):
        params = {'output': OUTPUT_CLASS,
                  'l_max_scalars': L_MAX_SCALARS,
                  'lensing': LENSING}
        params.update(cosmo_params)
        print(params)
        self.cosmo.set(params)
        self.cosmo.compute()
        cls = self.cosmo.lensed_cl(L_MAX_SCALARS)
        eb_tb = np.zeros(shape=cls["tt"].shape)
        _, Q, U = hp.synfast((cls['tt'], cls['ee'], cls['bb'], cls['te'], eb_tb, eb_tb), nside=self.NSIDE, new=True)
        self.cosmo.struct_cleanup()
        self.cosmo.empty()
        return Q, U

    def sample_mixing_matrix(self, betas):
        #mat_pixels = []
        #for i in range(self.Npix):
        #    m = self.mixing_matrix_evaluator(betas[i,:])[:, 1:]
        #    mat_pixels.append(m)

        mat_pixels = (self.mixing_matrix_evaluator(beta)[:, 1:] for beta in betas)
        return mat_pixels

    def sample_mixing_matrix_full(self, betas):
        #mat_pixels = []
        #for i in range(self.Npix):
        #    m = self.mixing_matrix_evaluator(betas[i,:])
        #    mat_pixels.append(m)

        mat_pixels = (self.mixing_matrix_evaluator(beta) for beta in betas)
        return mat_pixels

    def sample_model(self, input_params):
        random_seed = input_params
        np.random.seed(random_seed)
        cosmo_params, sampled_beta = self.sample_model_parameters()
        cosmo_dict = {l[0]: l[1] for l in zip(COSMO_PARAMS_NAMES, cosmo_params.tolist())}
        tuple_QU = self.sample_CMB_QU(cosmo_dict)
        map_CMB = np.concatenate(tuple_QU)
        result = {"map_CMB": map_CMB,"cosmo_params": cosmo_params,"betas": sampled_beta}
        with open("B3DCMB/data/temp" + str(random_seed), "wb") as f:
            pickle.dump(result, f)

        return cosmo_params

    def compute_weight(self, input):
        observed_data = config.sky_map
        noise_level, random_seed, means, sigmas_symm, denom = input
        np.random.seed(random_seed)
        with open("B3DCMB/data/temp" + str(random_seed), "rb") as f:
            data = pickle.load(f)

        map_CMB = data["map_CMB"]
        print("Duplicating CMB")
        duplicate_CMB = (l for l in map_CMB for _ in range(15))
        print("Splitting for computation")
        x = np.split((observed_data - np.array(duplicate_CMB)) - np.array(means), self.Npix*2)
        print("Computing log weights")
        r = -(1/2)*np.sum((np.dot(l[1], scipy.linalg.solve(l[0], l[1].T)) for l in zip(sigmas_symm, x)))
        lw = r + denom
        return lw

    def sample_data(self):
        print("Sampling parameters")
        cosmo_params, sampled_beta = self.sample_model_parameters()
        print("Computing mean and cov of map")
        mean_map = np.array([i for l in self.Qs + self.Us for i in l])
        stdd_map =[i for l in self.sigma_Qs + self.sigma_Us for i in l]
        print("Sampling maps Dust and Sync")
        maps = self.sample_normal(mean_map, stdd_map, diag = True)
        print("Computing cosmo params")
        cosmo_dict = {l[0]: l[1] for l in zip(COSMO_PARAMS_NAMES, cosmo_params.tolist())}
        print("Sampling CMB signal")
        tuple_QU = self.sample_CMB_QU(cosmo_dict)
        map_CMB = np.concatenate(tuple_QU)
        print("Creating mixing matrix")
        mixing_matrix = self.sample_mixing_matrix(sampled_beta)
        print("Scaling to frequency maps")
        #freq_maps = np.dot(scipy.linalg.block_diag(*2*mixing_matrix), maps.T)
        freq_pixels = []
        mix1, mix2 = tee(mixing_matrix)
        for i, mat in enumerate(chain(mix1, mix2)):
            freq_pix = np.dot(mat, maps[2*i:(2*i+2)].T)
            freq_pixels.append(freq_pix)

        freq_maps = np.concatenate(freq_pixels)
        print("Adding CMB to frequency maps")
        duplicated_cmb = np.repeat(map_CMB, 15)
        print("Creating noise")
        noise = self.sample_normal(np.zeros(2 * 15 * self.Npix), self.noise_stdd_all, diag = True)
        print("Adding noise to the maps")
        #sky_map_no_noise = freq_maps + duplicated_cmb
        sky_map = np.add(np.add(freq_maps, duplicated_cmb), noise)

        #sig = 1/(np.dot(np.dot(np.transpose(sky_map_no_noise), np.diag(1/(self.noise_stdd_all**2))), sky_map_no_noise))
        return duplicated_cmb
        #return {"sky_map": sky_map, "cosmo_params": cosmo_params, "betas": sampled_beta}
