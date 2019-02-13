from classy import Class

# Define your cosmology (what is not specified will be set to CLASS default parameters)
params = {
    'output': 'tCl pCl',
    'l_max_scalars': 5000,
    'lensing': 'no',
    'A_s': 2.3e-9,
    'n_s': 0.9624, 
    'h': 0.6711,
    'omega_b': 0.022068,
    'omega_cdm': 0.12029}#,
    # '':0.01}

# params['A_s'] = np.random.normal(average, dispersion)



# Create an instance of the CLASS wrapper
cosmo = Class()

# Set the parameters to the cosmological code
cosmo.set(params)

# Run the whole code. Depending on your output, it will call the
# CLASS modules more or less fast. For instance, without any
# output asked, CLASS will only compute background quantities,
# thus running almost instantaneously.
# This is equivalent to the beginning of the `main` routine of CLASS,
# with all the struct_init() methods called.
cosmo.compute()
print dir(cosmo)
# Access the lensed cl until l=2000
cls = cosmo.raw_cl(5000)
# cls = cosmo.lensed_cl(5000)

import healpy as hp
import pylab as pl
CMB = hp.synfast(cls['tt'], nside=128)
print CMB
print cls['tt']
hp.mollview(CMB)
pl.show()
exit()

# Print on screen to see the output
print cls
# It is a dictionnary that contains the fields: tt, te, ee, bb, pp, tp

# plot something with matplotlib...
import pylab as pl
for key in cls.keys():
	if key!='ell' and 'p' not in key : pl.loglog(cls['ell'], cls[key], label=key)
pl.legend()
pl.show()

# Clean CLASS (the equivalent of the struct_free() in the `main`
# of CLASS. This step is primordial when running in a loop over different
# cosmologies, as you will saturate your memory very fast if you ommit
# it.
cosmo.struct_cleanup()

# If you want to change completely the cosmology, you should also
# clean the arguments, otherwise, if you are simply running on a loop
# of different values for the same parameters, this step is not needed
cosmo.empty()