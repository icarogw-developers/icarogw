from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, is_there_cupy

class HI_map(object):
    def __init__(self,redshift_grid, pixel_grid, density_matrix):
        '''
        Ciao
        Parameters
        ----------
        redshift_grid: xp.array
            Redshift grid used to construct the density matrix
        pixel_grid: xp.array
            Pixel grid (healpy or mhealpy) indeces to construct the matrix
        density_matrix: xp.array
            Density Matrix 
        '''
        xp=get_module_array(redshift_grid)
        self.redshift_grid = redshift_grid
        self.pixel_grid = pixel_grid
        self.density_matrix = np.log(density_matrix) 
        self.density_matrix_average = np.log(xp.mean(density_matrix,axis=1)) # Check axis 

    def event_averaged_density(self,posterior_samples_catalog):

        sx=get_module_array_scipy(z)
        N_events = len(posterior_samples_catalog.posterior_samples_dict)
        events = list(posterior_samples_catalog.posterior_samples_dict.keys())

        self.list_PE_averaged_density = []
        for i in range(N_events):
            skyind = posterior_samples_catalog.posterior_samples_dict[events[i]].posterior_data['sky_indices']
            dm = np.vstack([np.exp(self.density_matrix[:,j]) 
                              for j in skyind])
            # Averaged over skymap
            avv = np.log(np.mean(dm,axis=0))

            self.list_PE_averaged_density.append(sx.interpolate.interp1d(self.redshift_grid,avv,kind='linear',bounds_error=False,
                                                  fill_value=-np.inf))

            

    def drho_dzdomega(self,z,skypos,cosmology,dl=None,average=False):
        '''
        Parameters
        ----------
        z: xp.array
            Redshift array
        skypos: xp.array
            Array containing the healpix indeces where to evaluate the interpolant (same indexing as grid interpolant)
        cosmology: class
            cosmology class to use for the computation
        dl: xp.array
            Luminosity distance in Mpc
        average: bool
            Use the sky averaged differential of effective number of galaxies in each pixel
        '''
        
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
            #interpolant = sx.interpolate.interp1d(z_grid,dNgal_dzdOm_sky_mean,kind='linear',fill_value='extrapolate')
            interpolant = sx.interpolate.interp1d(z_grid,dNgal_dzdOm_sky_mean,kind='linear',bounds_error=False,
                                                  fill_value=-np.inf) # If a posterior samples fall outside, then you return0
            gcpart=interpolant(z)
        else:
            gcpart=sx.interpolate.interpn((z_grid,pixel_grid),dNgal_dzdOm_vals,xp.column_stack([z,skypos]),bounds_error=False,
                                fill_value=-np.inf,method='linear') # If a posterior samples fall outside, then you return0
            #gcpart=sx.interpolate.interpn((z_grid,pixel_grid),dNgal_dzdOm_vals,xp.column_stack([z,skypos]),bounds_error=False,
            #                    fill_value=np.array([1e-10]),method='linear') # If a posterior samples fall outside, then you return0
        
        
        return gcpart.reshape(originshape)
