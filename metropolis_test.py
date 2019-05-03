import healpy as hp
from classy import Class


NSIDE = 512
Npix = 12 * NSIDE ** 2

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


