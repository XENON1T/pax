"""Very simple position reconstruction algorithms"""
from pax import plugin


from pax.datastructure import ReconstructedPosition


class PosRecWeightedSum(plugin.TransformPlugin):

    """Reconstruction using charge-weighted sum.

    Charge-weighted average of PMT positions.  Class to reconstruct s2 x, y
    positions using the weighted sum of PMTs in the top array.
    """

    def startup(self):
        self.top_array_map = self.config['topArrayMap']

    def transform_event(self, event):
        for peak in event.peaks:
            self.log.debug("Left %d, Right: %d" % (peak.left, peak.right))

            # This is an array where every i-th element is how many samples
            # were seen by the i-th PMT
            hits = event.pmt_waveforms[..., peak.left:peak.right].sum(axis=1)

            if hits.sum() != 0:
                sum_x = 0
                sum_y = 0

                for pmt, value in self.top_array_map.items():
                    x, y = value['x'], value['y']

                    x, y = hits[pmt] * x, hits[pmt] * y

                    sum_x += x
                    sum_y += y

                scale = hits.sum()

                peak_x = sum_x / scale
                peak_y = sum_y / scale
            else:
                peak_x = peak_y = float('NaN')  # algorithm failed

            rp = ReconstructedPosition({'x': peak_x,
                                        'y': peak_y,
                                        'z': float('nan'),
                                        'peak' : peak['index_of_maximum'],
                                        'algorithm': self.name})

            peak.reconstructed_positions.append(rp)

        return event
