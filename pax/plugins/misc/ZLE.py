import numpy as np

from pax import plugin, datastructure
from pax.plugins.signal_processing.HitFinder import find_intervals_above_threshold

import matplotlib.pyplot as plt

class SoftwareZLE(plugin.TransformPlugin):
    """Emulate the Zero-length encoding of the CAEN 1724 digitizer
    Makes no attempt to emulate the 2-sample word logic, so some rare edge cases will be different
    """
    debug = False

    def transform_event(self, event):

        new_pulses = []
        zle_intervals_buffer = -1 * np.ones((1000, 2), dtype=np.int64)

        for pulse in event.pulses:

            # Subtract the referemce baseline, invert
            # TODO: Does the ZLE know a better baseline? I guess not...
            w = pulse.raw_data.copy()
            w = self.config['digitizer_reference_baseline'] - w

            if self.debug:
                plt.plot(w)

            # Find intervals above ZLE threshold
            high_threshold = low_threshold = self.config['zle_threshold']
            n_itvs_found = find_intervals_above_threshold(w, high_threshold, low_threshold, zle_intervals_buffer)
            if n_itvs_found == 0:
                continue
            elif n_itvs_found == self.config['max_intervals']:
                # more than 1000 intervals - insane!!!
                zle_intervals_buffer[-1, 1] = pulse.length - 1
            itvs_to_encode = zle_intervals_buffer[:n_itvs_found]

            if self.debug:
                for l, r in itvs_to_encode:
                    plt.axvspan(l, r, alpha=0.5, color='red')

            # Find boundaries of regions to encode
            # This will introduce overlaps
            itvs_to_encode[:, 0] -= self.config['samples_to_store_before']
            itvs_to_encode[:, 1] += self.config['samples_to_store_after']

            # Decide which intervals to encode: have to deal with overlaps here
            itvs_encoded = 0
            itv_i = 0
            while itv_i <= len(itvs_to_encode) - 1:
                start = max(itvs_to_encode[itv_i, 0], 0)

                if itvs_encoded >= self.config['max_intervals']:
                    self.log.debug("ZLE breakdown in channel %d: all samples from %d onwards are stored" % (
                        pulse.channel, zle_intervals_buffer[-1, 0]))
                    stop = pulse.length - 1
                    itv_i = len(itvs_to_encode) - 1     # Loop will end after this last pulse is appended
                else:
                    stop = min(itvs_to_encode[itv_i, 1], pulse.length - 1)
                    # If next interval starts before this one ends, update stop and keep searching
                    # If last interval reached, there is no itv_i + 1, thats why the condition has <, not <=
                    while itv_i < len(itvs_to_encode) - 1:
                        if itvs_to_encode[itv_i + 1, 0] <= stop:
                            stop = min(itvs_to_encode[itv_i + 1, 1], pulse.length - 1)
                            itv_i += 1
                        else:
                            break

                if self.debug:
                    plt.axvspan(start, stop, alpha=0.3, color='green')

                new_pulses.append(datastructure.Pulse(
                    channel=pulse.channel,
                    left=pulse.left+start,
                    right=pulse.left+stop,
                    raw_data=pulse.raw_data[start:stop + 1]
                ))
                itvs_encoded += 1
                itv_i += 1

            if self.debug:
                plt.show()

        event.pulses = new_pulses
        return event