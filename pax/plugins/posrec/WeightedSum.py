import numpy as np
from pax import plugin


class PosRecWeightedSum(plugin.PosRecPlugin):
    """Reconstruct x,y positions as the charge-weighted average of PMT positions in the top array.
    """
    def reconstruct_position(self, peak):
        hitpattern = peak.area_per_channel[self.pmts]
        return np.average(self.pmt_locations, weights=hitpattern, axis=0)
