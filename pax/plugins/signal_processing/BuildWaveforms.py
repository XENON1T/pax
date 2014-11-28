import numpy as np
from pax import plugin, units, datastructure


class BuildWaveforms(plugin.TransformPlugin):

    def startup(self):
        c = self.config

        # Extract the number of PMTs from the configuration
        all_pmts = self.config['pmts_top'] | self.config['pmts_bottom'] | self.config['pmts_veto']
        self.n_pmts = len(all_pmts)
        if not max(all_pmts) == self.n_pmts-1:
            raise ValueError("PMT numbers should be an uninterrupted sequence starting from zero.")

        # Conversion factor from converting from ADC counts -> pmt-electrons/bin
        # Still has to be divided by PMT gain to get pe/bin
        self.conversion_factor = c['digitizer_t_resolution'] * c['digitizer_voltage_range'] / (
            2 ** (c['digitizer_bits']) * c['pmt_circuit_load_resistor']
            * c['external_amplification'] * units.electron_charge
        )

        # Channel groups starting with 'u' will be gain corrected using nominal gain
        self.channel_groups = {
            'top':              c['pmts_top'],
            'bottom':           c['pmts_bottom'],
            'veto':             c['pmts_veto'],
            's1_peakfinding':   (c['pmts_top'] | c['pmts_bottom']) - c['pmts_excluded_for_s1']
            # The 'tpc' wave will be added by summing 'top' and 'bottom'
        }
        if self.config['build_nominally_gain_corrected_waveforms']:
            # Also store nominal-gain corrected waveforms
            self.channel_groups.update({
                'uS1':  (c['pmts_top'] | c['pmts_bottom']) - c['pmts_excluded_for_s1'],
                'uS2':  (c['pmts_top'] | c['pmts_bottom']),
            })
        self.undead_channels = []


    def transform_event(self, event):

        # Sanity check
        if not self.config['digitizer_t_resolution'] == event.sample_duration:
            raise ValueError('Event %s quotes sample duration = %s ns, but digitizer_t_resolution is set to %s!' % (
                              event.event_number, event.sample_duration, self.config['digitizer_t_resolution']))

        # Initialize empty waveforms
        event.pmt_waveforms = np.zeros((self.n_pmts, event.length()))
        for group, members in self.channel_groups.items():
            event.waveforms.append(datastructure.Waveform({
                'samples':  np.zeros(event.length()),
                'name':     group,
                'pmt_list': self.crazy_type_conversion(members),
            }))

        for channel, waveform_occurrences in event.occurrences.items():

            # Check for unknown gains and undead channels
            if channel not in self.config['gains']:
                raise ValueError('Gain for channel %s is not specified!' % channel)
            if self.config['gains'][channel] == 0:
                if channel not in self.undead_channels:
                    if self.config['zombie_paranoia']:
                        self.log.warning('Undead channel %s: gain is set to 0, but it has a signal in this event!'  % channel)
                        self.log.warning('Further undead channel warnings for this channel will be suppressed.')
                    self.undead_channels.append(channel)
                if not self.config['build_nominally_gain_corrected_waveforms']:
                    # This channel won't add anything, so:
                    continue

            # Convert and store every occurrence in the right place
            baseline_sample = None
            for i, (start_index, occurrence_wave) in enumerate(waveform_occurrences):

                # Grab samples to compute baseline from

                # For pulses starting right after previous ones, we can keep the samples from the previous pulse
                if self.config['reuse_baseline_for_adjacent_occurrences'] \
                        and i > 0 \
                        and start_index == waveform_occurrences[i - 1][0] + len(waveform_occurrences[i - 1][1]):
                    #self.log.debug('Occurence %s in channel %s is adjacent to previous occurrence: reusing baseline' %
                    #               (i, channel))
                    pass

                # For VERY short pulses, we are in trouble...
                elif len(occurrence_wave) < self.config['baseline_sample_length']:
                    self.log.warning(
                        ("Occurrence %s in channel %s has too few samples (%s) to compute baseline:" +
                         ' reusing previous baseline in channel.') % (i, channel, len(occurrence_wave))
                    )
                    pass

                # For short pulses, we can take baseline samples from its rear.
                # The last occurrence is truncated in Xenon100, OK to use front-baselining even if short.
                elif self.config['rear_baselining_for_short_occurrences'] and  \
                            len(occurrence_wave) < self.config['rear_baselining_threshold_occurrence_length'] and \
                            (not start_index + len(occurrence_wave) > event.length() - 1):
                    if i > 0:
                        self.log.warning("Unexpected short occurrence %s in channel %s at %s (%s samples long)"
                                         % (i, channel, start_index, len(occurrence_wave)))
                    self.log.debug("Short pulse, using rear-baselining")
                    baseline_sample = occurrence_wave[len(occurrence_wave) - self.config['baseline_sample_length']:]

                # Finally, the usual baselining case:
                else:
                    baseline_sample = occurrence_wave[:self.config['baseline_sample_length']]

                # Compute the baseline from the baselining sample
                if baseline_sample is None:
                    self.log.warning(
                        ('DANGER: attempt to re-use baseline in channel %s where none has previously been computed: ' +
                         ' using default digitizer baseline %s.') %
                        (channel, self.config['digitizer_baseline'])
                    )
                    baseline = self.config['digitizer_baseline']
                else:
                    baseline = np.mean(baseline_sample)  # No floor, Xerawdp uses float arithmetic too. Good.

                # Truncate pulses starting too early
                if start_index < 0:
                    self.log.warning('Occurence %s in channel %s starts %s samples before event start: truncating.' % (
                        i, channel, -start_index
                    ))
                    occurrence_wave = (0, occurrence_wave[-start_index:])

                # Truncate pulses taking too long
                overhang_length = len(occurrence_wave) - 1 + start_index - event.length()
                if overhang_length > 0:
                    self.log.warning('Occurence %s in channel %s has overhang of %s samples: truncating.' % (
                        i, channel, overhang_length
                    ))
                    occurrence_wave = occurrence_wave[:len(occurrence_wave)-overhang_length]

                end_index = start_index + len(occurrence_wave)   # Well, not really index... index+1

                # Compute corrected pulse
                if self.config['build_nominally_gain_corrected_waveforms']:
                    nominally_corrected_pulse = (baseline - occurrence_wave) * self.conversion_factor / self.config['nominal_pmt_gain']
                    corrected_pulse = nominally_corrected_pulse * self.config['nominal_pmt_gain'] / self.config['gains'][channel]
                else:
                    corrected_pulse = (baseline - occurrence_wave) * self.conversion_factor / self.config['gains'][channel]

                # Store the waveform in pmt_waveforms, unless gain=0, then we leave it as 0
                # TODO: is this wise? How would we investigate undead channels if we don't store the data?
                if self.config['gains'][channel] != 0:
                    event.pmt_waveforms[channel][start_index:end_index] = corrected_pulse

                # Add corrected pulse to pmt_waveforms and all appropriate summed waveforms
                for group, members in self.channel_groups.items():
                    if channel in members:
                        if group[0] == 'u':
                            pulse_to_add = nominally_corrected_pulse
                        else:
                            pulse_to_add = corrected_pulse
                        event.get_waveform(group).samples[start_index:end_index] += pulse_to_add

        # Add the tpc waveform: sum of top and bottom
        event.waveforms.append(datastructure.Waveform({
            'samples':  event.get_waveform('top').samples + event.get_waveform('bottom').samples,
            'name':     'tpc',
            'pmt_list': self.crazy_type_conversion(self.channel_groups['top'] | self.channel_groups['bottom'])
        }))
        return event

    @staticmethod
    def crazy_type_conversion(x):
        return np.array(list(x), dtype=np.uint16)
