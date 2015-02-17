"""Very simple position reconstruction algorithms"""
from pax import plugin

from pax.datastructure import ReconstructedPosition


class PosRecWeightedSum(plugin.TransformPlugin):

    """Centroid reconstruction using a charge-weighted sum.

    Charge-weighted average of PMT positions.  Class to reconstruct S2's x, y
    positions using the weighted sum of PMTs in the top array.
    """

    def startup(self):
        """Initialize reconstruction algorithm

        Determine which PMTs to use for reconstruction.
        """

        # This can either be 'top' or 'bottom'.
        self.which_pmts = 'channels_%s' % self.config['channels_to_use_for_reconstruction']
        if self.which_pmts not in self.config.keys():
            raise RuntimeError("Bad choice 'channels_to_use_for_reconstruction'")

        # List of integers of which PMTs to use
        self.pmts = self.config[self.which_pmts]

        # (x,y) locations of these PMTs.  This is stored as a dictionary such
        # that self.pmt_locations[int] = {'x' : int, 'y' : int, 'z' : None}
        self.pmt_locations = self.config['pmt_locations']

    def transform_event(self, event):
        """Reconstruct the position of S2s in an event.

        Information on how to use the 'event' object is at:

          http://xenon1t.github.io/pax/format.html
        """

        # For every S2 peak found in the event
        for peak in event.S2s():
            # This is an array where every i-th element is how many pe
            # were seen by the i-th PMT
            hits = peak.area_per_channel

            sum_x = 0  # sum of x positions
            sum_y = 0  # sum of y positions

            scale = 0  # Total q

            # For every PMT
            for pmt in self.pmts:  # 'pmt' is a PMT ID
                value = self.pmt_locations[pmt]  # Dictionary {'x' : int, etc.}
                Q = hits[pmt]  # Area of this S2 for this PMT  # noqa

                # Add this 'Q' to total Q 'scale'
                scale += Q

                # Add charge-weighted position to running sum
                sum_x += Q * value['x']
                sum_y += Q * value['y']

            if scale != 0:  # If charge actually seen for this S2
                # Compute the positions
                peak_x = sum_x / scale
                peak_y = sum_y / scale
            else:
                peak_x = peak_y = float('NaN')  # algorithm failed

            # Create a reconstructed position object
            rp = ReconstructedPosition({'x': peak_x,
                                        'y': peak_y,
                                        'algorithm': self.name})

            # Append our reconstructed position object to the peak
            peak.reconstructed_positions.append(rp)

        # Return the event such that the next processor can work on it
        return event
