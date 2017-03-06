import numpy as np

import json

from pax import plugin, utils

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
        for i in range(len(r_pts)):  # unpacks 1d array to 2d
            aft_vals.append(data['map'][i*len(z_pts):(i+1)*len(z_pts)])
        aft_vals = np.array(aft_vals)
        self.aft_map = RectBivariateSpline(r_pts, z_pts, aft_vals)
        self.low_pe_threshold = 10  # below this in PE, transition to hits

    def transform_event(self, event):

        for ia in event.interactions:
            s1 = event.peaks[ia.s1]

            if s1.area < self.low_pe_threshold:
                s1_frac = s1.area/self.low_pe_threshold
                hits_top = s1.n_hits*s1.hit_fraction_top
                s1_top = s1.area*s1.area_fraction_top
                size_top = np.round(hits_top*(1-s1_frac) + s1_top*s1_frac)
                size_tot = np.round(s1.n_hits*(1-s1_frac) + s1.area*s1_frac)
            else:
                size_top = np.round(s1.area*s1.area_fraction_top)
                size_tot = np.round(s1.area)

            aft = self.aft_map(ia.r, ia.z)[0, 0]
            ia.s1_area_fraction_top_pvalue = binom_test(size_top, size_tot, aft)

        return event
