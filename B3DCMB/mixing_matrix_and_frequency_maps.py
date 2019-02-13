'''
Code for B3DCMB
1. loading CMB, dust and synchrotron templates (in uK_RJ at 150GHz)
2. Building mixing matrix
3. Generating frequency maps d = As
Author: Josquin
Contact: josquin@apc.in2p3.fr
'''

import sys
import os
import argparse
import numpy as np
import healpy as hp
import pylab as pl
import copy

###########################################################
# CONSTANTS
h_over_k =  0.0479924 # h/k
cst = 56.8 # cst for CMB = h/kT

###########################################################
'''
Parameters to be set up
- frequencies in GHz
- nside for the resolution of the maps
'''

frequencies = [30,40,90,150,220,270]
nside = 32
#### choose units between CMB-CMB or RJ-CMB depending on the input templates
units = 'CMB-CMB'
# units = 'RJ-CMB'

###########################################################
'''
Loading the sky templates, only the Q and U for CMB, dust and synchrotron
'''
# you can change the following path, where you store the maps
path2maps = './'
dustmap = 'dust_150GHz_uKRJ.fits'
syncmap = 'sync_150GHz_uKRJ.fits'
cmbmap = 'cmb_150GHz_uKRJ.fits'
# loading maps with healpy 
cmb = hp.read_map( os.path.join( path2maps, cmbmap), field=(0,1,2))
dust = hp.read_map( os.path.join( path2maps, dustmap), field=(0,1,2))
sync = hp.read_map( os.path.join( path2maps, syncmap), field=(0,1,2))
print cmb.shape
# update resolution of these to match the chosen nside
if hp.npix2nside(len(cmb.T))!= nside: cmb = hp.ud_grade(copy.deepcopy(cmb), nside_out=nside)
if hp.npix2nside(len(dust.T))!= nside: dust = hp.ud_grade(copy.deepcopy(dust), nside_out=nside)
if hp.npix2nside(len(sync.T))!= nside: sync = hp.ud_grade(copy.deepcopy(sync), nside_out=nside)
# convert them to CMB units 
def BB_factor_computation(nu):
	"""
	@brief: from CMB to RJ units, computed for a given frequency
	@return: CMB->RJ conversion factor
	"""
	BB_factor = (nu/cst)**2*np.exp(nu/cst)/(np.exp(nu/cst)-1)**2
	return BB_factor
if units=='CMB-CMB':
	dust /= BB_factor_computation(150.0)
	sync /= BB_factor_computation(150.0)
	cmb /= BB_factor_computation(150.0)
elif units == 'RJ-CMB': 
	print 'not changing the input templates, assumed to be in K_RJ'
else: 
	print 'you should define units to be RJ-CMB or CMB-CMB'
	exit()
# building sky signal
sky_signal = np.vstack((cmb[1:], dust[1:], sync[1:]))

###########################################################
'''
we define the spectral parameters, i.e. the free parameters that define the mixing matrix
'''
spectral_parameters_default = { 'nu_ref':150.0, 'Bd':1.59, 'Td':19.6, 'h_over_k':0.0479924,\
					 'drun':0.0, 'Bs':-3.1, 'srun':0.0, 'cst':56.8, 'nu_pivot_sync_curv':23.0 }

def tag_maker( stokes='Q', frequency=150.0):
	tag_f = stokes+str(frequency)+'GHz'	
	return tag_f

def A_matrix_builder( frequencies=[150.0], spectral_parameters=spectral_parameters_default):
	"""
	@brief: given a list of frequencies (in GHz) and a dictionnary containing the spectral parameters
	=> the function output a dictionnary containing the mixing matrix in A['matrix']
	"""
	# set the keys of the output mixing matrix
	A_output = {}#OrderedDict()
	A_output['out'] = []
	A_output['in'] = []

	# squeezed or not squeezed, that's the question
	frequency_stokes_loc = ['Q','U']
	stokes_temp_loc = ['Qcmb','Ucmb','Qdust','Udust','Qsync','Usync']

	# this sets the size of the mixing matrix 
	for f in frequencies:
		for f_stokes in frequency_stokes_loc:
			tag_f = tag_maker(stokes=f_stokes, frequency=f)			
			A_output['out'].append(tag_f)
	for stokes_temp in stokes_temp_loc:
		A_output['in'].append( stokes_temp )

	# setting the dimensions of the mixing matrix
	A_output['matrix'] = np.zeros((len(A_output['out']), len(A_output['in'])))

	# fill in the mixing matrix
	for freq_loc in frequencies:
		for f_stokes in frequency_stokes_loc:

			tag_f = tag_maker(stokes=f_stokes, frequency=freq_loc)			
			indo = A_output['out'].index(tag_f)
			bandpass = [1.0]
			freq_range = [freq_loc]

			# integration over bandpasses 
			for i_nu in range(len(freq_range)):
				spectral_parameters['nu'] = freq_range[i_nu]*1.0
				# numerical estimation of A
				nu = spectral_parameters['nu']*1.0
				cst = spectral_parameters['cst']*1.0
				nu_ref = spectral_parameters['nu_ref']*1.0
				Bd = spectral_parameters['Bd']*1.0
				Td = spectral_parameters['Td']*1.0
				h_over_k = spectral_parameters['h_over_k']*1.0
				drun = spectral_parameters['drun']*1.0
				Bs = spectral_parameters['Bs']*1.0
				srun = spectral_parameters['srun']*1.0
				if 'nu_pivot_sync_curv' in spectral_parameters.keys():
					nu_pivot_sync_curv = spectral_parameters['nu_pivot_sync_curv']*1.0
				else: nu_pivot_sync_curv = 23.0

				##############################
				# CMB component
				Qcmb_LOC = (nu / cst) ** 2 * ( np.exp ( nu / cst ) ) / ( ( np.exp ( nu / cst ) - 1 ) ** 2 )
				if units=='CMB-CMB':
					Qcmb_LOC *= BB_factor_computation(150.0)/BB_factor_computation(nu)
				Ucmb_LOC = Qcmb_LOC*1.0
				##############################
				# dust component
				Qdust_LOC = ( np.exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( np.exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * np.log( nu/nu_ref ) )
				if units=='CMB-CMB':
					Qdust_LOC *= BB_factor_computation(150.0)/BB_factor_computation(nu)
				Udust_LOC = Qdust_LOC*1.0
				##############################
				# synchrotron
				Qsync_LOC = ( nu / nu_ref ) ** (Bs + srun * np.log( nu / nu_pivot_sync_curv ) )
				if units=='CMB-CMB':
					Qsync_LOC *= BB_factor_computation(150.0)/BB_factor_computation(nu)

				Usync_LOC = Qsync_LOC*1.0
				##############################
				A_line = []
				A_line.append(Qcmb_LOC)
				A_line.append(Ucmb_LOC)
				A_line.append(Qdust_LOC)
				A_line.append(Udust_LOC)
				A_line.append(Qsync_LOC)
				A_line.append(Usync_LOC)
				############################
				conversion_RJ2CMB = 1.0
				############################
				if i_nu == 0:
					A_bandpass = bandpass[i_nu] * np.array( A_line ) #/ conversion_RJ2CMB
				else:
					A_bandpass += bandpass[i_nu] * np.array( A_line ) #/ conversion_RJ2CMB
				############################
				del Qcmb_LOC, Ucmb_LOC, Qdust_LOC, Udust_LOC, Qsync_LOC, Usync_LOC

			if freq_range[-1]==freq_range[0]:
				A_output['matrix'][indo,:] = A_bandpass / np.sum( bandpass )
			else:
				A_output['matrix'][indo,:] = A_bandpass*(freq_range[1]-freq_range[0])/(freq_range[-1]-freq_range[0])

			del A_bandpass

	# setting to zero off-diagonal (Q,U) elements
	for stokes_temp in stokes_temp_loc:
		for freq_loc in frequencies:
			for f_stokes in frequency_stokes_loc:
				if (f_stokes not in stokes_temp):
					tag_f = tag_maker(stokes=f_stokes, frequency=freq_loc)			
					indo = A_output['out'].index(tag_f)
					indi = A_output['in'].index(stokes_temp)
					A_output['matrix'][indo,indi] = 0.0
	return A_output

# computation of the mixing matrix
Mixing_Matrix = A_matrix_builder( frequencies=frequencies)

###########################################################
'''
Creation of the frequency maps, i.e. d = As
'''
frequency_maps = Mixing_Matrix['matrix'].dot( sky_signal ) 

N_freq = int(len(frequency_maps)/2.0)
for f in range(N_freq):
	hp.mollview( frequency_maps[2*f,:], sub=(N_freq,2,2*f+1), title='Q map @ = '+str(frequencies[f])+' GHz', unit=r'$\mu$K$_{\rm CMB}$')
	hp.mollview( frequency_maps[2*f+1,:], sub=(N_freq,2,2*f+2), title='U map @ = '+str(frequencies[f])+' GHz', unit=r'$\mu$K$_{\rm CMB}$')
pl.show()



