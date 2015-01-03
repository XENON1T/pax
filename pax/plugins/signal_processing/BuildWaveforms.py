import numpy as np

from pax import plugin, units, datastructure


class BuildWaveforms(plugin.TransformPlugin):
    """

    Waveforms that will be built:
        tpc
        top
        bottom
        NAME for each external detector named NAME (usually only the veto)
    for Xerawdp matching also:
        uS1
        uS2
    """

    def startup(self):
        c = self.config

        # Extract the number of PMTs from the configuration
        self.n_pmts = self.config['n_pmts']

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
            's1_peakfinding':   (c['pmts_top'] | c['pmts_bottom']) - c['pmts_excluded_for_s1']
        }
        # Add each detector as a channel group
        for name, chs in c['channels_in_detector'].items():
            self.channel_groups[name] = chs
        self.external_detectors = [k for k in c['channels_in_detector'].keys() if k != 'tpc']

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
        for group, members in self.channel_groups.items():
            event.waveforms.append(datastructure.SumWaveform({
                'samples':  np.zeros(event.length()),
                'name':     group,
                'pmt_list': np.array(list(members), dtype=np.uint16),
                'detector': group if group in self.external_detectors else 'tpc'
            }))

        last_occurrence_in = {}

        for occ_i, occ in enumerate(event.occurrences):

            channel = occ.channel

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

            baseline_sample = None
            start_index = occ.left
            length = occ.length
            end_index = occ.right
            occurrence_wave = occ.raw_data

            # Grab samples to compute baseline from

            # For pulses starting right after previous ones, we can keep the samples from the previous pulse
            if self.config['reuse_baseline_for_adjacent_occurrences'] \
                    and channel in last_occurrence_in \
                    and start_index == last_occurrence_in[channel].right + 1:
                #self.log.debug('Occurence %s in channel %s is adjacent to previous occurrence: reusing baseline' %
                #               (i, channel))
                pass

            # For VERY short pulses, we are in trouble...
            elif length < self.config['baseline_sample_length']:
                self.log.warning(
                    ("Occurrence %s in channel %s has too few samples (%s) to compute baseline:" +
                     ' reusing previous baseline in channel.') % (occ_i, channel, length)
                )
                pass

            # For short pulses, we can take baseline samples from its rear.
            # The last occurrence is truncated in Xenon100, OK to use front-baselining even if short.
            elif self.config['rear_baselining_for_short_occurrences'] and  \
                        len(occurrence_wave) < self.config['rear_baselining_threshold_occurrence_length'] and \
                        (not start_index + length > event.length() - 1):
                if occ_i > 0:
                    self.log.warning("Unexpected short occurrence %s in channel %s at %s (%s samples long)"
                                     % (occ_i, channel, start_index, length))
                self.log.debug("Short pulse, using rear-baselining")
                baseline_sample = occurrence_wave[length - self.config['baseline_sample_length']:]

            # Finally, the usual baselining case:
            else:
                baseline_sample = occurrence_wave[:self.config['baseline_sample_length']]

            # Compute the baseline from the baselining sample
            if baseline_sample is None:
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
                baselining_method = self.config.get('find_baselines_using', 'mean')
                if baselining_method == 'mean':
                    # Xerawdp behaviour
                    baseline = np.mean(baseline_sample)  # No floor, Xerawdp uses float arithmetic too. Good.
                elif baselining_method == 'median':
                    # More robust against peaks in start of sample
                    baseline = np.median(baseline_sample)
                else:
                    raise ValueError("Invalid find_baselines_using: should be 'mean' or 'median'")

            # Throw error if occurrence is completely outside event -- see issue 43
            if end_index < 0 or start_index > event.length() - 1:
                raise ValueError('Occurrence %s in channel %s (%s-%s) is entirely outside event bounds (%s-%s)!' % (
                    occ_i, channel, start_index, end_index, 0, event.length()-1))

            # Truncate occurrences starting too early -- see issue 43
            if start_index < 0:
                self.log.warning(
                    'Occurence %s in channel %s starts %s samples before event start: truncating. See issue #43.' % (
                    occ_i, channel, -start_index))
                occurrence_wave = occurrence_wave[-start_index:]
                # Update the start index
                start_index = 0

            # Truncate occurrences taking too long -- see issue 43
            overhang_length = len(occurrence_wave) - 1 + start_index - event.length()
            if overhang_length > 0:
                self.log.warning(
                    'Occurrence %s in channel %s has overhang of %s samples: truncating. See issue #43.' % (
                    occ_i, channel, overhang_length))
                occurrence_wave = occurrence_wave[:length-overhang_length]
                # Update the length & end index
                length = len(occurrence_wave)
                end_index = start_index + length - 1

            # Compute corrected pulse
            if self.config['build_nominally_gain_corrected_waveforms']:
                nominally_corrected_pulse = (baseline - occurrence_wave) * self.conversion_factor / self.config['nominal_pmt_gain']
                corrected_pulse = nominally_corrected_pulse * self.config['nominal_pmt_gain'] / self.config['gains'][channel]
            else:
                corrected_pulse = (baseline - occurrence_wave) * self.conversion_factor / self.config['gains'][channel]

            # Store the waveform in pmt_waveforms, unless gain=0, then we leave it as 0
            # TODO: is this wise? How would we investigate undead channels if we don't store the data?
            if self.config['gains'][channel] != 0:
                event.pmt_waveforms[channel][start_index:end_index + 1] = corrected_pulse

            # Add corrected pulse to pmt_waveforms and all appropriate summed waveforms
            for group, members in self.channel_groups.items():
                if channel in members:
                    if group[0] == 'u':
                        pulse_to_add = nominally_corrected_pulse
                    else:
                        pulse_to_add = corrected_pulse
                    event.get_waveform(group).samples[start_index:end_index + 1] += pulse_to_add

            # Store some metadata for this occurrence
            occ.height = np.max(corrected_pulse)
            occ.digitizer_baseline_used = baseline

            last_occurrence_in[channel] = occ

        return event