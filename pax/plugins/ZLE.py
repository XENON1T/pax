import numpy as np

from pax import plugin, datastructure

from pax.dsputils import find_intervals_above_threshold_no_splitting

import matplotlib.pyplot as plt


class SoftwareZLE(plugin.TransformPlugin):
    """Emulate the Zero-length encoding of the CAEN 1724 digitizer
    Makes no attempt to emulate the 2-sample word logic, so some rare edge cases will be different

    Uses a separate debug setting, as need to show plots
    """
    debug = False
    zle_intervals_buffer = -1 * np.ones((50000, 2), dtype=np.int64)

    def transform_event(self, event):
        new_pulses = []
        zle_intervals_buffer = self.zle_intervals_buffer

        for pulse_i, pulse in enumerate(event.pulses):
            if self.debug:
                print("Starting ZLE in pulse %d, channel %d" % (pulse_i, pulse.channel))
            if pulse.left % 2 != 0:
                raise ValueError("Cannot ZLE in XED-compatible way "
                                 "if pulse starts at odd sample index (%d)" % pulse.left)

            w = pulse.raw_data.copy()

            samples_for_baseline = self.config.get('initial_baseline_samples', None)
            if samples_for_baseline is not None:
                # This tries to do better than the digitizer: compute a baseline for the pulse
                bs = w[:min(len(w), samples_for_baseline)]
                w = np.mean(bs) - w
            else:
                # This is how the digitizer does it (I think???)
                # Subtract the reference baseline, invert
                w = self.config['digitizer_reference_baseline'] - w

            if self.debug:
                plt.plot(w)

            # Get the ZLE threshold
            # Note a threshold of X digitizer bins actually means that the data acquisition
            # triggers when the waveform becomes greater than X, i.e. X+1 or more (see #273)
            # hence the + 1
            if str(pulse.channel) in self.config.get('special_thresholds', {}):
                threshold = self.config['special_thresholds'][str(pulse.channel)] + 1
            else:
                threshold = self.config['zle_threshold'] + 1

            # Find intervals above ZLE threshold
            # We need to call the version with numba boost
            n_itvs_found = find_intervals_above_threshold_no_splitting(w.astype(np.float64),
                                                          threshold=threshold,
                                                          result_buffer=zle_intervals_buffer,
                                                          )

            if n_itvs_found == self.config['max_intervals']:
                # more than 5000 intervals - insane!!!
                # Ignore intervals beyond this -- probably will go beyond 32 intervals to encode anyway
                zle_intervals_buffer[-1, 1] = pulse.length - 1

            if n_itvs_found > 0:
                itvs_to_encode = zle_intervals_buffer[:n_itvs_found]

                if self.debug:
                    for l, r in itvs_to_encode:
                        plt.axvspan(l, r, alpha=0.5, color='red')

                # Find boundaries of regions to encode by subtracting before and after window
                # This will introduce overlaps and out-of-pulse indices
                itvs_to_encode[:, 0] -= self.config['samples_to_store_before']
                itvs_to_encode[:, 1] += self.config['samples_to_store_after']

                # Clip out-of-pulse indices
                itvs_to_encode = np.clip(itvs_to_encode, 0, pulse.length - 1)

                # Decide which intervals to encode: deal with overlaps here
                itvs_encoded = 0
                itv_i = 0
                while itv_i <= len(itvs_to_encode) - 1:
                    start = itvs_to_encode[itv_i, 0]

                    if itvs_encoded >= self.config['max_intervals']:
                        self.log.debug("ZLE breakdown in channel %d: all samples from %d onwards are stored" % (
                            pulse.channel, zle_intervals_buffer[-1, 0]))
                        stop = pulse.length - 1
                        itv_i = len(itvs_to_encode) - 1     # Loop will end after this last pulse is appended
                    else:
                        stop = itvs_to_encode[itv_i, 1]
                        # If next interval starts before this one ends, update stop and keep searching
                        # If last interval reached, there is no itv_i + 1, thats why the condition has <, not <=
                        while itv_i < len(itvs_to_encode) - 1:
                            if itvs_to_encode[itv_i + 1, 0] <= stop:
                                stop = itvs_to_encode[itv_i + 1, 1]
                                itv_i += 1
                            else:
                                break

                    # Truncate the interval to the nearest even start and odd stop index
                    # We use truncation rather than extension to ensure data always exists
                    # pulse.left is guaranteed to be even
                    if start % 2 != 0:
                        start += 1
                    if stop % 2 != 1:
                        stop -= 1
                    assert (stop - start + 1) % 2 == 0

                    if self.debug:
                        plt.axvspan(start, stop, alpha=0.3, color='green')

                    # Explicit casts necessary since we've disabled type checking for pulse class
                    # for speed in event builder.
                    # and otherwise numpy ints would get in and break e.g. BSON output.
                    new_pulses.append(datastructure.Pulse(
                        channel=int(pulse.channel),
                        left=int(pulse.left+start),
                        right=int(pulse.left+stop),
                        raw_data=pulse.raw_data[start:stop + 1]
                    ))
                    itvs_encoded += 1
                    itv_i += 1

            if self.debug:
                plt.show()

        event.pulses = new_pulses
        return event
