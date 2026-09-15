from scipy.interpolate import interpn
import numpy as np

class density_contrast_map(object):
    def __init__(self,dcenters,density_contrast_matrix,map):
        '''
        dcenters: np.array
        density_contrast: (realization,dcom,sky_index)
        map: mhealp map used for pixelization
        '''

        self.dcenters = dcenters
        self.density_contrast_matrix = density_contrast_matrix
        self.map = map
        self.pixgrid = np.arange(self.map.npix)

    def get_density_contrast(self,dcom_h,skyindex):

        return interpn((self.dcenters,self.pixgrid),
                                      self.density_contrast_matrix,
                                      np.column_stack([dcom_h,skyindex]), 
                                      bounds_error=False, fill_value=0.,method='splinef2d') # If a posterior sample falls outside, then you return 0