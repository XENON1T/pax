import numpy as np
from pax import plugin, units

class JoinAndConvertWaveforms(plugin.TransformPlugin):

    """Take channel_occurrences, builds channel_waveforms

    Between occurrence waveforms (i.e. pulses...), zeroes are added.
    The waveforms returned will be converted to pe/ns and baseline-corrected.
    If a channel is absent from channel_occurrences, it wil be absent from channel_waveforms.

    """

    def startup(self):
        # Short hand
        c = self.config
        self.gains = self.config['gains']

        # Conversion factor from converting from ADC counts -> pe/bin
        self.conversion_factor = c['digitizer_t_resolution'] * c['digitizer_voltage_range'] / (
            2 ** (c['digitizer_bits']) * c['pmt_circuit_load_resistor']
            * c['external_amplification'] * units.electron_charge
        )

    def transform_event(self, event):
        # Check if voltage range same as reported by input plugin
        # TODO: do the same for dt
        # TODO: move to XED input plugin
        # if 'metadata' in event and 'voltage_range' in event['metadata']:
        #     if event['metadata']['voltage_range'] != self.config['digitizer_voltage_range']:
        #         raise RuntimeError(
        #             'Voltage range from event metadata (%s) is different from ini file setting (%s)!'
        #             % (event['metadata']['voltage_range'], self.config['digitizer_voltage_range'])
        #         )

        # Check for input plugin misbehaviour / running this plugin at the wrong time
        # if not ('channel_occurrences' in event and 'event_duration' in event):
        #     raise RuntimeError(
        #         "Event contains %s, should contain at least channel_occurrences and event_duration !"
        #         % str(event.keys())
        #     )

        # Dump digests of channels included
        # bla = list(map(int,event['channel_occurrences'].keys()))
        # print(np.sum(bla), np.sum(np.log(bla)))

        # Build the channel waveforms from occurrences
        # event['processed_waveforms'] = {}
        uncorrected_sum_wave_for_s1 = np.zeros(event.length())
        uncorrected_sum_wave_for_s2 = np.zeros(event.length())
        pmt_waveform_matrix = np.zeros((999,event.length())) #TODO: get max pmt number somewhere
        # event['channel_waveforms']   = {}
        baseline_sample_size = 46 #TODO: put in config!!!!
        for channel, waveform_occurrences in event.occurrences.items():
            skip_channel = False  # Temp for Xerawdp matching, refactor to continue's later

            # Check that gain known
            if channel not in self.gains:
                self.log.warning('Gain for channel %s is not specified! Skipping channel.' % channel)
                skip_channel = True

            # Deal with unknown gains
            if self.gains[channel] == 0:
                if channel in event.occurrences:
                    self.log.debug('Gain for channel %s is 0, but is in waveform.' % channel)
                skip_channel = True

            # Assemble the waveform pulse by pulse, starting from an all-zeroes waveform
            wave = np.zeros(event.length())

            for i, (starting_position, wave_occurrence) in enumerate(waveform_occurrences):

                # Check for pulses starting right after previous ones: Xerawdp doesn't recompute baselines
                if i !=0 and starting_position == waveform_occurrences[i-1][0]+len(waveform_occurrences[i-1][1]):
                    pass #baseline will still have the right value
                else:
                    # We need to compute the baseline.
                    # Only pulses at the end and beginning of the trace are allowed to be shorter than 2*46
                    # In case of a short first pulse, computes baseline from its last samples instead of its first.
                    if (
                        not (starting_position + len(wave_occurrence) > event.length()-1) and
                        len(wave_occurrence) < 2*baseline_sample_size
                    ):
                        if i != 0:
                            raise RuntimeError("Occurrence %s in channel %s at %s has event_duration %s, should be at least 2*%s!"
                                               % (i, channel, starting_position, len(wave_occurrence), baseline_sample_size)
                            )
                        self.log.debug("Short first pulse, computing baseline from its LAST samples")
                        baseline_sample = wave_occurrence[len(wave_occurrence)-baseline_sample_size:]
                    else:
                        baseline_sample = wave_occurrence[:baseline_sample_size]
                    baseline = np.mean(baseline_sample)  # No floor, Xerawdp uses float arithmetic. Good.
                    """
                    This is NOT THE WAY TO DO IT - we should at least average over all occurrences
                    Better yet, take a mean of median 20% or so:
                        return (
                            np.mean(sorted(baseline_sample)[
                                    int(0.4 * len(baseline_sample)):int(0.6 * len(baseline_sample))
                                    ]),  # Ensures peaks in baseline sample don't skew computed baseline
                            np.std(baseline_sample)  # ... but do count towards baseline_stdev!
                        )
                    Don't want to just take the median as V-resolution is finite
                    Don't want the mean either: this is not robust against large fluctuations (eg peaks in sample)
                    Better yet, do this for several events, keep a running mean
                    However, this is how Xerawdp does it... (see Rawdata.cpp, getPulses)
                    # TODO: Check for baseline fluctuations in event, warn if too much
                    # How much baseline will we have in 1T? Only few samples?
                    """

                corrected_pulse = baseline - wave_occurrence   # Note: flips up!

                # Put wave occurrences in the correct positions

                # Temp for Xerawdp matching: add pulse to the uncorrected sum waveform if they are not excluded
                if not channel > 178:
                    uncorrected_sum_wave_for_s2[starting_position:starting_position + len(wave_occurrence)] += corrected_pulse
                    if not channel in self.config['pmts_excluded_for_s1']:
                        uncorrected_sum_wave_for_s1[starting_position:starting_position + len(wave_occurrence)] += corrected_pulse

                if skip_channel: continue
                wave[starting_position:starting_position + len(wave_occurrence)] = corrected_pulse

            if skip_channel: continue
            # Convert wave to pe/ns, and store it in the event data structure
            pmt_waveform_matrix[channel] = wave * self.conversion_factor / self.gains[channel]

        # Store everything
        event.pmt_waveforms = pmt_waveform_matrix
        # Temp for Xerawdp matching: store uncorrected sum waveform
        universal_gain_correction = self.conversion_factor / (2*10**6)
        event.append_waveform(
            samples=uncorrected_sum_wave_for_s1 * universal_gain_correction,
            name='uS1',
            pmt_list=set(list(range(1,178))) - self.config['pmts_excluded_for_s1'],
        )
        event.append_waveform(
            samples=uncorrected_sum_wave_for_s2 * universal_gain_correction,
            name='uS2',
            pmt_list=set(list(range(1,178))),
        )
        # TODO: Maybe Delete the channel_occurrences from the event structure, we don't need it anymore

        return event


class SumWaveforms(plugin.TransformPlugin):

    """Build the sum waveforms for, top, bottom, top_and_bottom, veto

    Since channel waveforms are already gain corrected, we can just add the appropriate channel waveforms.
    If none of the channels in a group contribute, the summed waveform will be all zeroes.
    This guarantees that e.g. event['processed_waveforms']['top_and_bottom'] exists.

    """

    def startup(self):
        self.channel_groups = {'top': self.config['pmts_top'],
                               'bottom': self.config['pmts_bottom'],
                               'veto': self.config['pmts_veto']}

        # The groups are lists, so we add them using |, not +...
        self.channel_groups['top_and_bottom'] = (self.channel_groups['top'] | self.channel_groups['bottom'])
        self.channel_groups['top_and_bottom_for_s1'] = (self.channel_groups['top'] | self.channel_groups['bottom']) - self.config['pmts_excluded_for_s1']
        # TEMP for XerawDP matching: Don't have to compute peak finding waveform yet, done in JoinAndConvertWaveforms

    def transform_event(self, event):
        # Compute summed waveforms
        for group, members in self.channel_groups.items():
            event.append_waveform(samples=np.sum(event.pmt_waveforms[[list(members)]], axis=0), name=group, pmt_list=members)

        return event


