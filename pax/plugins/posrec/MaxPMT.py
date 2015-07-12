import numpy as np

from pax import plugin
from pax.datastructure import ReconstructedPosition


class PosRecMaxPMT(plugin.TransformPlugin):

    """Reconstruct S2 x,y positions at the PMT in the top array that shows the largest signal (in area)
    """

    def startup(self):
        if self.config['pmt_0_is_fake']:
            self.input_channels = np.array(self.config['channels_top'][1:])
        else:
            self.input_channels = np.array(self.config['channels_top'])

    def transform_event(self, event):
        """Reconstruct the position of S2s in an event.
        """

        # For every S2 peak found in the event
        for peak in event.S2s():

            input_areas = peak.area_per_channel[self.input_channels]

            max_pmt = self.input_channels[np.argmax(input_areas)]

            peak.reconstructed_positions.append(ReconstructedPosition({
                'x': self.config['pmt_locations'][max_pmt]['x'],
                'y': self.config['pmt_locations'][max_pmt]['y'],
                'algorithm': self.name}))

        return event
