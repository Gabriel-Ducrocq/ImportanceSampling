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

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
COSMO_PARAMS_SIGMA = [0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]
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
        self.cosmo_var = (np.diag(COSMO_PARAMS_SIGMA)/2)**2

        self.instrument = pysm.Instrument(get_instrument('litebird', self.NSIDE))
        self.components = [CMB(), Dust(150.), Synchrotron(150.)]
        self.mixing_matrix = MixingMatrix(*self.components)
        self.mixing_matrix_evaluator = self.mixing_matrix.evaluator(self.instrument.Frequencies)
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

    def sample_normal(self, mu, sigma, s = None):
        return np.random.multivariate_normal(mu, sigma, s)

    def sample_model_parameters(self):
        sampled_cosmo = self.sample_normal(self.cosmo_means, self.cosmo_var)
        sampled_beta = self.sample_normal(self.matrix_mean, np.diag(self.matrix_var)).reshape((self.Npix, -1), order = "F")
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

    def sample_model(self, observed_data):
        cosmo_params, sampled_beta = self.sample_model_parameters()
        cosmo_dict = {l[0]: l[1] for l in zip(COSMO_PARAMS_NAMES, cosmo_params.tolist())}
        tuple_QU = self.sample_CMB_QU(cosmo_dict)

        map_CMB = np.concatenate(tuple_QU)
        mixing_matrix = self.sample_mixing_matrix(sampled_beta)

        all_mixing_matrix = 2*mixing_matrix
        means_and_sigmas = [[np.dot(l[0], l[1]), np.diag([1e-10 for _ in range(15)]) + np.einsum("ij,jk,lk", l[0], np.diag(l[2]), l[0])]
            for l in zip(all_mixing_matrix, self.Qs + self.Us, self.sigma_Qs + self.sigma_Us)]
        means, sigmas = zip(*means_and_sigmas)
        sigmas = [(s+s.T)/2 for s in sigmas]
        inv_sigmas= [scipy.linalg.inv(s) for s in sigmas]
        mean = np.array([i for l in means for i in l])
        duplicate_CMB = np.array([l for l in map_CMB for _ in range(15)])
        x = np.split((observed_data - duplicate_CMB) - mean, 24)
        log_det = np.sum([np.log(scipy.linalg.det(2*np.pi*s)) for s in sigmas])
        denom = -(1 / 2) * log_det
        lw = -(1/2)*np.sum([np.dot(l[1], scipy.linalg.solve(l[0], l[1].T)) for l in zip(inv_sigmas, x)]) + denom
        return {"map_CMB": map_CMB,"cosmo_params": cosmo_params,"betas": sampled_beta,"log_weight": lw}

    def sample_data(self):
        cosmo_params, sampled_beta = self.sample_model_parameters()
        mean_map = [i for l in self.Qs + self.Us for i in l]
        covar_map =[i for l in self.sigma_Qs + self.sigma_Us for i in l]
        maps = self.sample_normal(mean_map, np.diag(covar_map))
        cosmo_dict = {l[0]: l[1] for l in zip(COSMO_PARAMS_NAMES, cosmo_params.tolist())}
        tuple_QU = self.sample_CMB_QU(cosmo_dict)
        map_CMB = np.concatenate(tuple_QU)
        mixing_matrix = self.sample_mixing_matrix(sampled_beta)
        freq_maps = np.dot(scipy.linalg.block_diag(*2*mixing_matrix), maps.T)
        duplicated_cmb = np.array([l for l in map_CMB for _ in range(15)])
        sky_map = freq_maps + duplicated_cmb
        return {"sky_map": sky_map, "cosmo_params": cosmo_params, "betas": sampled_beta}
