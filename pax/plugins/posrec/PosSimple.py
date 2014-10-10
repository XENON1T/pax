"""Very simple position reconstruction algorithms"""
from pax import plugin

from pax.datastructure import ReconstructedPosition


class PosRecWeightedSum(plugin.TransformPlugin):
    """Reconstruction using charge-weighted sum.

    Charge-weighted average of PMT positions.  Class to reconstruct s2 x, y
    positions using the weighted sum of PMTs in the top array.
    """

    def startup(self):
        self.which_pmts = self.config['pmts_to_use_for_reconstruction']

        if self.which_pmts == 'top' or self.which_pmts == 'bottom':
            self.pmts = self.config['pmts_%s' % self.which_pmts]
        else:
            raise RuntimeError("Bad choice 'pmts_to_use_for_reconstruction'")

        self.pmt_locations = self.config['pmt_locations']

    def transform_event(self, event):
        for peak in event.peaks:
            self.log.debug("Left %d, Right: %d" % (peak.left, peak.right))

            # This is an array where every i-th element is how many samples
            # were seen by the i-th PMT
            hits = peak.area_per_pmt

            if hits.sum() != 0:
                sum_x = 0  # sum of x positions
                sum_y = 0  # sum of y positions

                scale = 0  # Total q

                for pmt in self.pmts:
                    value = self.pmt_locations[pmt]
                    Q = hits[pmt]

                    scale += Q

                    sum_x += Q * value['x']
                    sum_y += Q * value['y']

                peak_x = sum_x / scale
                peak_y = sum_y / scale
            else:
                peak_x = peak_y = float('NaN')  # algorithm failed

            rp = ReconstructedPosition({'x': peak_x,
                                        'y': peak_y,
                                        'z': float('nan'),
                                        'index_of_maximum': peak.index_of_maximum,
                                        'algorithm': self.name})

            peak.reconstructed_positions.append(rp)

        return event
