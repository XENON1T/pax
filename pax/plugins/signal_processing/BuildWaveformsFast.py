import numpy as np
import numba

from pax import plugin, units, datastructure, exceptions


class BuildWaveformsFast(plugin.TransformPlugin):
    """Construct sum waveforms from PMT channel pulses
    """

    def startup(self):
        self.undead_channels = []
        c = self.config

        # Extract the number of PMTs from the configuration
        self.n_channels = self.config['n_channels']
        self.truncate_occurrences_partially_outside = self.config.get('truncate_occurrences_partially_outside', False)

        # Conversion factor from converting from ADC counts -> pmt-electrons/bin
        # Still has to be divided by PMT gain to get pe/bin
        self.conversion_factor = 2 ** (c['digitizer_bits'])
        self.conversion_factor *= c['pmt_circuit_load_resistor']
        self.conversion_factor *= c['external_amplification']
        self.conversion_factor *= units.electron_charge
        self.conversion_factor = 1 / self.conversion_factor  # Invert!
        self.conversion_factor *= c['sample_duration']
        self.conversion_factor *= c['digitizer_voltage_range']

        # Build the channel -> detector lookup dict
        self.detector_by_channel = {}
        for name, chs in c['channels_in_detector'].items():
            for ch in chs:
                self.detector_by_channel[ch] = name


    def transform_event(self, event):
        # Sanity check
        if not self.config['sample_duration'] == event.sample_duration:
            raise ValueError('Event %s quotes sample duration = %s ns, but sample_duration is set to %s!' % (
                event.event_number, event.sample_duration, self.config['sample_duration']))

        event_length = event.length()

        # Initialize empty waveforms for each detector
        for detector, chs in self.config['channels_in_detector'].items():
            event.sum_waveforms.append(datastructure.SumWaveform(
                samples=np.zeros(event_length),
                name=detector,
                channel_list=np.array(list(chs), dtype=np.uint16),
                detector=detector
            ))

        last_occurrence_in = {}

        for occ_i, occ in enumerate(event.occurrences):

            channel = occ.channel
            detector = self.detector_by_channel[channel]
            start_index = occ.left
            length = occ.length
            end_index = occ.right

            # Ignore dead channels
            if self.config['gains'][channel] == 0:
                continue

            occurrence_wave = occ.raw_data

            ##
            #  Pulse bounds checking / truncation
            ##

            # Always throw error if occurrence is completely outside event -- see issue 43
            if end_index < 0 or start_index > event_length - 1:
                raise exceptions.OccurrenceBeyondEventError(
                    'Occurrence %s in channel %s (%s-%s) is entirely outside event bounds (%s-%s)!' % (
                        occ_i, channel, start_index, end_index, 0, event_length - 1))

            # Truncate occurrences starting too early -- see issue 43
            if start_index < 0:
                message_head = "Occurrence %s in channel %s starts %s samples before event starts (see issue #43)" % (
                    occ_i, channel, -start_index)
                if not self.truncate_occurrences_partially_outside:
                    raise exceptions.OccurrenceBeyondEventError(message_head + '!')
                self.log.warning(message_head + ': truncating.')
                occurrence_wave = occurrence_wave[-start_index:]
                # Update the start index
                start_index = 0

            # Truncate occurrences taking too long -- see issue 43
            overhang_length = length - 1 + start_index - event_length
            if overhang_length > 0:
                message_head = "Occurrence %s in channel %s has overhang of %s samples (see issue #43)" % (
                    occ_i, channel, overhang_length)
                if not self.truncate_occurrences_partially_outside:
                    raise exceptions.OccurrenceBeyondEventError(message_head + '!')
                self.log.warning(message_head + ': truncating.')
                occurrence_wave = occurrence_wave[:length - overhang_length]
                # Update the length & end index
                length = len(occurrence_wave)
                end_index = start_index + length - 1

            ##
            #  Baseline determination
            ##

            # For pulses starting right after previous ones, we may want to keep the baseline
            reuse_baseline = self.config['reuse_baseline_for_adjacent_occurrences'] \
                and channel in last_occurrence_in \
                and start_index == last_occurrence_in[channel].right + 1

            # For very short pulses, we should certainly keep our previous baseline
            reuse_baseline = reuse_baseline or (length < self.config['baseline_sample_length'])

            if reuse_baseline:
                if channel in last_occurrence_in:
                    baseline = last_occurrence_in[channel].digitizer_baseline_used
                else:
                    self.log.warning(
                        ('DANGER: attempt to re-use baseline in channel %s where none has previously been computed: ' +
                         ' using default digitizer baseline %s.') %
                        (channel, self.config['digitizer_baseline'])
                    )
                    baseline = self.config['digitizer_baseline']

            else:
                baseline_sample = occurrence_wave[:self.config['baseline_sample_length']]
                baseline = self._fast_mean(baseline_sample)

            ##
            # Apply baseline and gain correction, waveform summation
            ##

            # Compute corrected pulse
            corrected_pulse = (baseline - occurrence_wave) * self.conversion_factor / self.config['gains'][channel]

            # Store the waveform in channel_waveforms
            event.channel_waveforms[channel][start_index:end_index + 1] = corrected_pulse

            # Add corrected pulse to the summed waveform
            event.get_sum_waveform(detector).samples[start_index:end_index + 1] += corrected_pulse

            # Store some metadata for this occurrence
            occ.height = self._fast_max(corrected_pulse)
            occ.digitizer_baseline_used = baseline

            last_occurrence_in[channel] = occ

        return event

    @staticmethod
    @numba.jit(numba.float64(numba.int16[:]))
    def _fast_mean(w):
        mean = 0
        for x in w:
            mean += x
        mean /= len(w)
        return mean

    @staticmethod
    @numba.jit(numba.float64(numba.float64[:]))
    def _fast_max(w):
        current_max = -999.9
        for x in w:
            if x > current_max:
                current_max = x
        return current_max
