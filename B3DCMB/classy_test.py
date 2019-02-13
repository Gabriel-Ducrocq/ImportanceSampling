from classy import Class
import healpy as hp
import numpy as np

params = [
    ('output', 'tCl pCl lCl'),
    ('l_max_scalars', 5000),
    ('lensing', 'yes'),
    ('n_s', 0.9665),
    ('omega_b', 0.02242),
    ('omega_cdm', 0.11933),
    ('100*theta_s', 1.04101),
    ('ln10^{10}A_s', 3.047),
    ('tau_reio', 0.0561)]




cosmo = Class()
cosmo.set(params)
cosmo.compute()
cls = cosmo.lensed_cl(5000)
cls_list = [it for _, it in cls.items()]
eb_tb = np.zeros(shape=cls["tt"].shape)
I,Q,U = hp.synfast((cls['tt'],cls['ee'],cls['bb'],cls['te'],eb_tb,eb_tb), nside=32, new = True)
print(Q.shape)
cosmo.struct_cleanup()
cosmo.empty()

