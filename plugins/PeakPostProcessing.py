"""Post processing for peaks.

Should include any plugins run directly after peak finding to prep information
needed for more complex transforms.
"""
import numpy as np

from pax import plugin


class MakeHitList(plugin.TransformPlugin):
    """Make hit lists.

    Class to make a hit list for each s1 and s2 peak as well as the
    multiplicity. Hit list stored in list with event_duration equal to number of
    PMTs."""

    def startup(self):
        self.num_channels = self.config['num_pmts']

    def transform_event(self, event):
        for peak in event.peaks:
            pmt_list = np.zeros(self.num_channels,
                                np.int16)

            for i, waveform in enumerate(self.pmt_waveforms):
                hit_pmts = []

                if waveform[peak.left:peak.right].sum() > 0:
                    hit_pmts.append(i)

            peak.pmt_list = pmt_list

        return event
