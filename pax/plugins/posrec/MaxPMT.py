import numpy as np
from pax import plugin


class PosRecMaxPMT(plugin.PosRecPlugin):
    """Reconstruct x,y positions at the PMT in the top array that shows the largest signal (in area)
    """
    def reconstruct_position(self, peak):
        max_pmt = np.argmax(peak.area_per_channel[self.pmts])
        return self.pmt_locations[max_pmt]
