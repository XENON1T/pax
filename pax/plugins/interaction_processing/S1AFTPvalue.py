import numpy as np

import json

from pax import plugin, exceptions
from pax.datastructure import Interaction

from scipy.interpolate import RectBivariateSpline
from scipy.stats import binom_test

class S1AreaFractionTopPvalue(plugin.TransformPlugin):
    """Computes p-value for S1 area fraction top
    
    """

    def startup(self):
        aftmap_filename = utils.data_file_name('s1_aft_rz_02Mar2017.json')
        with open(aftmap_filename) as data_file:
            data = json.load(data_file)
        r_pts = np.array(data['r_pts'])
        z_pts = np.array(data['z_pts'])
        aft_vals = []
        for i in range(len(r_pts)): # unpacks 1d array to 2d
            aft_vals.append(data['map'][i*len(z_pts):(i+1)*len(z_pts)])
        aft_vals = np.array(aft_vals)[0,0]
        self.aft_map = RectBivariateSpline(r_pts,z_pts,aft_vals)

    def transform_event(self, event):

        for ia in event.interactions:
            s1 = event.peaks[ia.s1]
            s2 = event.peaks[ia.s2]
            
            s1_top = np.round(s1.area*s1.area_fraction_top)
            s1_tot = np.round(s1.area)
            aft = self.aft_map(np.sqrt(ia.x**2 + ia.y**2),ia.z)
            ia.pvalue_s1_area_fraction_top = binom_test(s1_top, s1_tot, aft)

        return event
