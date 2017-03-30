import numpy as np

from pax import plugin, utils
from pax.InterpolatingMap import InterpolatingMap

from scipy.stats import binom_test


class S1AreaFractionTopProbability(plugin.TransformPlugin):
    """Computes p-value for S1 area fraction top
    """

    def startup(self):
        aftmap_filename = utils.data_file_name('s1_aft_xyz_XENON1T_06Mar2017.json')
        self.aft_map = InterpolatingMap(aftmap_filename)
        self.low_pe_threshold = 10  # below this in PE, transition to hits
        self.max_s1_area = 1e4  # above this, don't bother

    def transform_event(self, event):

        for ia in event.interactions:
            s1 = event.peaks[ia.s1]

            if s1.area > self.max_s1_area:
                continue

            if s1.area < self.low_pe_threshold:
                s1_frac = s1.area/self.low_pe_threshold
                hits_top = s1.n_hits*s1.hits_fraction_top
                s1_top = s1.area*s1.area_fraction_top
                size_top = np.round(hits_top*(1-s1_frac) + s1_top*s1_frac)
                size_tot = np.round(s1.n_hits*(1-s1_frac) + s1.area*s1_frac)
            else:
                size_top = np.round(s1.area*s1.area_fraction_top)
                size_tot = np.round(s1.area)

            aft = self.aft_map.get_value(ia.x, ia.y, ia.z)
            ia.s1_area_fraction_top_probability = binom_test(size_top, size_tot, aft)

        return event
