from .cupy_pal import get_module_array, get_module_array_scipy

class density_contrast_map(object):
    def __init__(self,dcenters,density_contrast_matrix,map):
        '''
        dcenters: np.array
        density_contrast: (realization,dcom,sky_index)
        map: mhealp map used for pixelization
        '''
        
        xp=get_module_array(dcenters)
        self.dcenters = dcenters
        self.density_contrast_matrix = density_contrast_matrix
        self.map = map
        self.pixgrid = xp.arange(self.map.npix)

    def get_density_contrast(self,dcom_h,skyindex):
        xp=get_module_array(dcom_h)
        sx=get_module_array_scipy(dcom_h)
        return sx.interpolate.interpn((self.dcenters,self.pixgrid),
                                      self.density_contrast_matrix,
                                      xp.column_stack([dcom_h,skyindex]), 
                                      bounds_error=False, fill_value=0.,method='splinef2d') # If a posterior sample falls outside, then you return 0