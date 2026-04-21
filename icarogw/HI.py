from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, is_there_cupy

class HI_map(object):
    def __init__(self,redshift_grid, pixel_grid, density_matrix, bGW, alphaGW):
        """
    Interpolated (1 + HI-density contrast) map on a 2D grid of redshift and pixel index.

        matrix_raw(z, pix) = 1 + delta_HI(z, pix)
        delta_HI(z, pix) = [counts(z, pix) - average_counts(z)] / average_counts(z)

    The class stores a raw (1 + HI-density contrast) matrix and applies a redshift-dependent bias model:

        bias(z) = bGW * (1 + z)**alphaGW

    The biased density contrast map is then constructed as:

        density_matrix(z, pix) = bias(z) * (matrix_raw(z, pix) - 1) + 1

    and clipped to a small positive floor for numerical stability.

    Parameters
    ----------
    redshift_grid : xp.ndarray
        1D redshift grid used to define the first axis of `density_matrix`.
    pixel_grid : xp.ndarray
        1D pixel-index grid used to define the second axis of `density_matrix`.
    density_matrix : xp.ndarray
        2D raw density matrix with shape (n_z, n_pix).
    bGW : float
        Amplitude of the redshift-dependent bias.
    alphaGW : float
        Power-law index of the redshift-dependent bias.

    """
        
        xp=get_module_array(redshift_grid)
        self.redshift_grid = redshift_grid
        self.pixel_grid = pixel_grid
        self.density_matrix_raw = density_matrix
        self.population_parameters = ['bGW', 'alphaGW']
        self.update(bGW=bGW, alphaGW=alphaGW)

    # new function to take into account the bias dependence
    def update(self, **kwargs):
        """
        Update the population bias parameters and rebuild the biased density matrix.
        """
        xp = get_module_array(self.redshift_grid)
        self.biasGW = kwargs['bGW']
        self.biasGWpower = kwargs['alphaGW']
        bias = self.biasGW * (1.0 + self.redshift_grid) ** self.biasGWpower
        self.density_matrix = np.clip(bias[:, None] * (self.density_matrix_raw - 1) + 1, 1e-5, None)
        self.density_matrix_average = xp.mean(self.density_matrix, axis=1)


    def drho_dzdomega(self,z,skypos,cosmology,dl=None,average=False):
        """
        Evaluate the interpolated density field at given redshifts and sky positions.

        Parameters
        ----------
        z: xp.array
            Redshift array
        skypos: xp.array
            Array containing the healpix indices where to evaluate the interpolant (same indexing as grid interpolant)
        cosmology: class
            cosmology class to use for the computation
        dl: xp.array
            Luminosity distance in Mpc
        average: bool
            If True, use the sky-averaged density as a function of redshift only.
            If False, interpolate on the full (redshift, pixel) grid.

        Notes
        -----
        - Out-of-bounds values are assigned a fill value of 1.0 (density contrast = 0)
        - Linear interpolation is used.
        """
        
        xp=get_module_array(z)
        sx=get_module_array_scipy(z)
        
        originshape=z.shape
        z=z.flatten()
        skypos=skypos.flatten()
        
        if dl is None:
            dl=cosmology.z2dl(z)
        dl=dl.flatten()
        
        z_grid = self.redshift_grid
        dNgal_dzdOm_sky_mean = self.density_matrix_average
        dNgal_dzdOm_vals = self.density_matrix
        pixel_grid = self.pixel_grid
                
        if average:
            interpolant = sx.interpolate.interp1d(z_grid,dNgal_dzdOm_sky_mean,kind='linear',bounds_error=False, fill_value=1.)
            gcpart=interpolant(z)
        else:
            gcpart=sx.interpolate.interpn((z_grid,pixel_grid),dNgal_dzdOm_vals,xp.column_stack([z,skypos]),bounds_error=False,
                                          fill_value=1.,method='linear')
        
        return gcpart.reshape(originshape)
