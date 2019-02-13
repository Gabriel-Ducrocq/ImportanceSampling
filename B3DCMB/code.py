import numpy as np
import scipy 
import healpy as hp
import pylab as pl
import fgbuster
from fgbuster.observation_helpers import get_instrument, get_sky
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix
import pysm
import sys
import scipy

# make figures
# make_plot = True
make_plot = False

# define resolution of the maps
NSIDE=32#512
Npix = 12*NSIDE**2
# define the sky templates 
sky=pysm.Sky(get_sky(NSIDE,'d0s0'))
# dust and synchrotron templates @ 150GHz
dust = sky.dust(150)
sync = sky.synchrotron(150)


'''
############################################
#### CONSTRUCTING TEMPLATES AND COVARIANCE
#### -> p(t) \propto exp[-1/2 (t-\bar{t})^T N_t^{-1} (t-bar{t})]
############################################
# covariance for the templates
# synchrotron template @ 30GHz
sync_map_ = hp.read_map('COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits', field=(0,1,2,3,4,5))
sync_map_ = hp.ud_grade(sync_map_, nside_out=NSIDE)
Q_sync = sync_map_[0]
U_sync = sync_map_[1]
sigma_Q_sync = sync_map_[4] - sync_map_[2]
sigma_U_sync = sync_map_[5] - sync_map_[3]
# dust template @ 353GHz
dust_map_ = hp.read_map('COM_CompMap_DustPol-commander_1024_R2.00.fits', field=(0,1,2,3,4,5))
dust_map_ = hp.ud_grade(dust_map_, nside_out=NSIDE)
Q_dust = dust_map_[0]
U_dust = dust_map_[1]
sigma_Q_dust = dust_map_[4] - dust_map_[2]
sigma_U_dust = dust_map_[5] - dust_map_[3]

templates_map = np.hstack((Q_sync.T, U_sync.T, Q_dust.T, U_dust.T))
sigma_templates = [sigma_Q_sync, sigma_U_sync, sigma_Q_dust, sigma_U_dust]
covariance_templates = scipy.linalg.block_diag(*[np.diag(s_) for s_ in sigma_templates])

# showing these maps
if make_plot:
	hp.mollview(Q_sync, sub=(241), title='Q sync')
	hp.mollview(U_sync, sub=(242), title='U sync')
	hp.mollview(Q_dust, sub=(243), title='Q dust')
	hp.mollview(U_dust, sub=(244), title='U dust')
	hp.mollview(sigma_Q_sync, sub=(245), title='sigma Q sync')
	hp.mollview(sigma_U_sync, sub=(246), title='sigma U sync')
	hp.mollview(sigma_Q_dust, sub=(247), title='sigma Q dust')
	hp.mollview(sigma_U_dust, sub=(248), title='sigma U dust')
	pl.show()
	# exit()
def p_template_computation(temp_):
	logprob = -0.5*np.sum(templates_map**2/np.diag(covariance_templates)**2)
	return logprob
# build a random vector of template of size 4 x Npix 
# for example : 
template_test = np.random.normal(0,1,(4,Npix))
print p_template_computation(template_test)
'''
############################################
####  CONSTRUCTING THE MIXING MATRIX A
############################################
# define the instrumental specifications
instrument = pysm.Instrument(get_instrument(NSIDE, 'litebird'))
# define the components in the sky and their scaling laws
components=[CMB(), Dust(150.), Synchrotron(150.)]
# initiate function to estimate scaling laws for LiteBIRD frequencies
A = MixingMatrix(*components)
A_ev = A.evaluator(instrument.Frequencies)
print(A_ev([1.56,20,-3.1]).shape)
print(instrument.Frequencies)

'''
####################################################
####  CONSTRUCTING TEMPLATES OF BETA AND SIGMA BETA
#### -> p(\beta} \propto exp[-1/2 (\beta-\bar{\beta})^T \sigma_\beta^{-2} (\beta-\bar{\beta})]
####################################################
dust_spectral_indices_ = hp.read_map('COM_CompMap_dust-commander_0256_R2.00.fits', field=(3,5,6,8))
dust_spectral_indices_ = hp.ud_grade(dust_spectral_indices_, nside_out=NSIDE)
beta_dust = dust_spectral_indices_[2]
sigma_beta_dust = dust_spectral_indices_[3]
temp_dust = dust_spectral_indices_[0]
sigma_temp_dust = dust_spectral_indices_[1]
beta_sync = hp.read_map('sync_beta.fits', field=(0))
beta_sync = hp.ud_grade(beta_sync, nside_out=NSIDE)
sigma_beta_sync = np.random.normal(0,0.1,beta_sync.shape)
def p_beta_computation(beta_):
	logprob = -0.5*np.sum((beta_[0] - beta_dust)**2/sigma_beta_dust**2)\
			-0.5*np.sum((beta_[1] - temp_dust)**2/sigma_temp_dust**2 )\
			  -0.5*np.sum((beta_[2] - beta_sync)**2/sigma_sync_dust**2)
	return logprob
# build a beta vector for sampling, of size 3 x Npix 
# for example : 
beta_test = np.random.normal(0,1,(3,Npix))
print p_beta_computation(beta_test)
'''
exit()