"""
XerawdpImitation - XerawdpImitation Is Not Xerawdp
This code imitates the peakfinding of Xerawdp (the Xenon100 processor)
Its only purpose is to show we understand Xerawdp enough to be able to reproduce it,
it is not meant to replace Xerawdp, nor to be used unmodified by Xenon1T or any experiment.
(for example, some Xerawdp bugs are included here on purpose, to better match the Xerawdp output)
"""
import numpy as np
import logging

from scipy.signal import butter, filtfilt

from pax import plugin, datastructure, exceptions, units


class BuildWaveforms(plugin.TransformPlugin):
    """Construct waveforms from PMT channel pulses
    This code has been used for other stuff in pax for a while, that's why there's some stuff not related to Xerawdp

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
        self.undead_channels = []
        c = self.config

        # Extract the number of PMTs from the configuration
        self.n_channels = self.config['n_channels']
        self.truncate_pulses_partially_outside = self.config.get('truncate_pulses_partially_outside', False)

        # Conversion factor from converting from ADC counts -> pmt-electrons/bin
        # Still has to be divided by PMT gain to get pe/bin
        self.conversion_factor = 2 ** (c['digitizer_bits'])
        self.conversion_factor *= c['pmt_circuit_load_resistor']
        self.conversion_factor *= c['external_amplification']
        self.conversion_factor *= units.electron_charge
        self.conversion_factor = 1 / self.conversion_factor  # Invert!
        self.conversion_factor *= c['sample_duration']
        self.conversion_factor *= c['digitizer_voltage_range']

        self.channel_groups = {
            'top':              c['channels_top'],
            'bottom':           c['channels_bottom'],
        }

        # Add each detector as a channel group
        for name, chs in c['channels_in_detector'].items():
            self.channel_groups[name] = chs

        self.external_detectors = [k for k in c['channels_in_detector'].keys() if k != 'tpc']

        # For Xerawdp matching: have to exclude some channels from S1 peakfinding and build sum wv with nominal gain
        self.xerawdp_matching = c.get('xerawdp_matching', False)
        if self.xerawdp_matching:
            s1_channels = list(
                set(c['channels_top']) | set(c['channels_bottom']) - set(c['channels_excluded_for_s1']))
            self.channel_groups.update({
                's1_peakfinding': s1_channels,
                'uS1':  s1_channels,
                'uS2':  c['channels_top'] + c['channels_bottom'],
            })

    def transform_event(self, event):
        # Sanity check
        if not self.config['sample_duration'] == event.sample_duration:
            raise ValueError('Event %s quotes sample duration = %s ns, but sample_duration is set to %s!' % (
                event.event_number, event.sample_duration, self.config['sample_duration']))

        # Initialize empty waveforms
        for group, members in self.channel_groups.items():
            event.sum_waveforms.append(datastructure.SumWaveform(
                samples=np.zeros(event.length()),
                name=group,
                channel_list=np.array(list(members), dtype=np.uint16),
                detector=group if group in self.external_detectors else 'tpc'
            ))

        # DIRTY HACK, DON'T TELL CHRIS
        # Sneakily add channel_waveforms, no longer in our datastructure...
        object.__setattr__(event, 'channel_waveforms',
                           np.zeros((self.config['n_channels'], event.length()), dtype=np.float64))

        last_pulse_in = {}

        for occ_i, occ in enumerate(event.pulses):

            channel = occ.channel

            # Check for undead channels
            if self.config['gains'][channel] == 0:
                if channel not in self.undead_channels:
                    if self.config['zombie_paranoia']:
                        self.log.warning('Undead channel %s: gain is set to 0, '
                                         'but it has a signal in this event!' % channel)
                        self.log.warning('Further undead channel warnings for this channel will be suppressed.')
                    self.undead_channels.append(channel)
                if not self.xerawdp_matching:
                    # This channel won't add anything, so:
                    continue

            baseline_sample = None
            start_index = occ.left
            length = occ.length
            end_index = occ.right
            pulse_wave = occ.raw_data

            # Grab samples to compute baseline from

            # For pulses starting right after previous ones, we can keep the samples from the previous pulse
            if self.config['reuse_baseline_for_adjacent_pulses'] \
                    and channel in last_pulse_in \
                    and start_index == last_pulse_in[channel].right + 1:
                # self.log.debug('Occurence %s in channel %s is adjacent to previous pulse: reusing baseline' %
                #               (i, channel))
                pass

            # For VERY short pulses, we are in trouble...
            elif length < self.config['baseline_sample_length']:
                self.log.warning(
                    ("Pulse %s in channel %s has too few samples (%s) to compute baseline:" +
                     ' reusing previous baseline in channel.') % (occ_i, channel, length)
                )
                pass

            # For short pulses, we can take baseline samples from its rear.
            # The last pulse is truncated in Xenon100, OK to use front-baselining even if short.
            elif self.config['rear_baselining_for_short_pulses'] and  \
                    len(pulse_wave) < self.config['rear_baselining_threshold_pulse_length'] and \
                    (not start_index + length > event.length() - 1):
                if occ_i > 0:
                    self.log.warning("Unexpected short pulse %s in channel %s at %s (%s samples long)"
                                     % (occ_i, channel, start_index, length))
                self.log.debug("Short pulse, using rear-baselining")
                baseline_sample = pulse_wave[length - self.config['baseline_sample_length']:]

            # Finally, the usual baselining case:
            else:
                baseline_sample = pulse_wave[:self.config['baseline_sample_length']]

            # Compute the baseline from the baselining sample
            if baseline_sample is None:
                if channel in last_pulse_in:
                    baseline = last_pulse_in[channel].baseline
                else:
                    self.log.warning(
                        ('DANGER: attempt to re-use baseline in channel %s where none has previously been computed: ' +
                         ' using default digitizer baseline %s.') %
                        (channel, self.config['digitizer_reference_baseline'])
                    )
                    baseline = self.config['digitizer_reference_baseline']
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

            # Always throw error if pulse is completely outside event -- see issue 43
            if end_index < 0 or start_index > event.length() - 1:
                raise exceptions.PulseBeyondEventError(
                    'Pulse %s in channel %s (%s-%s) is entirely outside event bounds (%s-%s)!' % (
                        occ_i, channel, start_index, end_index, 0, event.length() - 1))

            # Truncate pulses starting too early -- see issue 43
            if start_index < 0:
                message_head = "Pulse %s in channel %s starts %s samples before event starts (see issue #43)" % (
                    occ_i, channel, -start_index)
                if not self.truncate_pulses_partially_outside:
                    raise exceptions.PulseBeyondEventError(message_head + '!')
                self.log.warning(message_head + ': truncating.')
                pulse_wave = pulse_wave[-start_index:]
                # Update the start index
                start_index = 0

            # Truncate pulses taking too long -- see issue 43
            overhang_length = len(pulse_wave) - 1 + start_index - event.length()
            if overhang_length > 0:
                message_head = "Pulse %s in channel %s has overhang of %s samples (see issue #43)" % (
                    occ_i, channel, overhang_length)
                if not self.truncate_pulses_partially_outside:
                    raise exceptions.PulseBeyondEventError(message_head + '!')
                self.log.warning(message_head + ': truncating.')
                pulse_wave = pulse_wave[:length - overhang_length]
                # Update the length & end index
                length = len(pulse_wave)
                end_index = start_index + length - 1

            # Compute corrected pulse
            if self.xerawdp_matching:
                nominally_corrected_pulse = baseline - pulse_wave
                nominally_corrected_pulse *= self.conversion_factor / self.config['nominal_gain']
                corrected_pulse = nominally_corrected_pulse
                if self.config['gains'][channel] == 0:
                    corrected_pulse = float('inf')
                else:
                    corrected_pulse *= self.config['nominal_gain'] / self.config['gains'][channel]
            else:
                corrected_pulse = (baseline - pulse_wave) * self.conversion_factor / self.config['gains'][channel]

            # Store the waveform in channel_waveforms
            # If the gain is 0, we can still be here if self.xerawdp_matching, don't want to store infs, so check first
            if self.config['gains'][channel] != 0:
                event.channel_waveforms[channel][start_index:end_index + 1] = corrected_pulse

            # Add corrected pulse to all appropriate summed waveforms
            for group, members in self.channel_groups.items():
                if channel in members:
                    if self.xerawdp_matching and group.startswith('u'):
                        pulse_to_add = nominally_corrected_pulse
                    else:
                        pulse_to_add = corrected_pulse
                    event.get_sum_waveform(group).samples[start_index:end_index + 1] += pulse_to_add

            # Store some metadata for this pulse
            occ.maximum = np.max(corrected_pulse)
            occ.baseline = baseline

            last_pulse_in[channel] = occ

        return event


class FindPeaks(plugin.TransformPlugin):

    """NB: Does NOT do veto peakfinding!!!"""

    def transform_event(self, event):

        # Load settings here, so we are sure dynamic threshold stuff etc gets reset
        # Don't need to move these to ini I think, this plugin is meant to match Xerawdp, which always used these
        self.settings_for_peaks = [   # We need a specific order, so we can't use a dict
            ('large_s2', {
                'threshold':                        0.6241506363,
                'left_boundary_to_height_ratio':    0.005,
                'right_boundary_to_height_ratio':   0.002,
                # +1 as Xerawdp tests for ... > minwidth, which is stupid, so my code doesn't
                'min_length':                       35 + 1,
                'max_length':                       float('inf'),
                'min_base_interval_length':         60 + 1,
                'max_base_interval_length':         float('inf'),
                'source_waveform':                  'filtered_for_large_s2',
                # For isolation test on top-level interval
                'around_interval_to_height_ratio_max': 0.05,
                'test_around_interval':             21,
                # For isolation test on every peak
                'around_peak_to_height_ratio_max':  0.25,
                'test_around_peak':                 21,
                'min_crossing_length':              1,
                'stop_if_start_exceeded':           False,  # Not for large s2s? Bug?
            }),
            ('small_s2', {
                'threshold':                        0.06241506363,
                'left_boundary_to_height_ratio':    0.01,
                'right_boundary_to_height_ratio':   0.01,
                'min_base_interval_length':         40 + 1,
                'max_base_interval_length':         200 - 1,
                'source_waveform':                  'filtered_for_small_s2',
                'around_to_height_ratio_max':       0.05,
                'test_around':                      10,
                'min_crossing_length':              1,
                'stop_if_start_exceeded':           True,
                'aspect_ratio_threshold':           0.06241506363,  # height/width = 1 mV/samples
                'stop_after_s2_height':             624.150636300,
            }),
            ('s1', {
                'threshold':                        0.1872451909,
                'left_boundary_to_height_ratio':    0.005,
                'right_boundary_to_height_ratio':   0.005,
                'min_crossing_length':              3,
                'test_before':                      50,
                'test_after':                       10,
                'before_avg_max_ratio':             0.01,
                'after_avg_max_ratio':              0.04,
                'stop_if_start_exceeded':           True,
                'source_waveform':                 'uS1',
                'max_filtered_width':               50 - 1,
                'stop_after_s2_height':             3.1207531815,
                # The s1 candidate interval isn't tested:
                'min_base_interval_length':         0,
                'max_base_interval_length':         float('inf'),
            }),
        ]
        self.seeker_position = None
        self.highest_s2_height_ever = 0
        for (peak_type, settings) in self.settings_for_peaks:
            # Get the signal out
            signal = event.get_sum_waveform(settings['source_waveform']).samples

            # Delete peaks beyond the 32 with largest area - must do AFTER small s2 peakfinding
            if peak_type == 's1':
                event.peaks = sort_and_prune_by(
                    event.peaks,
                    key=lambda x: getattr(x, 'area'),
                    keep_number=(32 if self.config['prune_peaks'] else float('inf')),
                    reverse=True
                )

            ##
            # STAGE 1 - REGION FINDING
            ##
            # Determines the regions in which to search for peak candidate intervals
            ##

            # Determine when we should stop looking for this type of peaks
            stop_looking_after = float('inf')
            if peak_type == 'small_s2':
                # For small s2s, we stop looking after a sufficiently large s2 (height in large_s2 waveform) is seen
                # Don't have to test these peaks are actually s2s, those are the only peaks in here
                stop_looking_after = min(
                    [stop_looking_after] +
                    [p.left for p in event.peaks if p.height > settings['stop_after_s2_height']]
                )
            if peak_type == 's1':
                if event.peaks:
                    # We stop looking for s1s after the s2 with the largest area
                    # undocumented!
                    stop_looking_after = [event.peaks[0].left]

                    # Also stop looking after s2s with large enough amplitude.
                    # Xerawdp redetermines the amplitude here, using the s1
                    # peakfinding waveform!
                    for p in event.peaks:
                        wf = event.get_sum_waveform('uS1')
                        height = wf.samples[p.index_of_maximum]
                        if height > settings['stop_after_s2_height']:
                            stop_looking_after.append(p.left)

                    stop_looking_after = min(stop_looking_after)

            self.log.debug("Starting %s search, stop looking after %s" % (peak_type, stop_looking_after))

            # Find the free regions - regions where peaks haven't yet been found
            # We could move this to the event class...
            lefts = sorted([0] + [p.left for p in event.peaks])
            rights = sorted([p.right for p in event.peaks] + [event.length() - 1])
            # Assuming each peak's right > left, we can simply split sorted(lefts+rights) in pairs:
            free_regions = list(zip(*[iter(sorted(lefts + rights))] * 2))
            self.log.debug("Free regions: " + str(free_regions))

            # Construct search regions from the free regions
            search_regions = []
            for region_l, region_r in free_regions:
                if region_l >= stop_looking_after:
                    break
                # S1s have a protection against starting too soon
                # (I guess S2s don't need this because of the filters applied to the waveform)
                if peak_type == 's1' and region_l < 12:
                    region_l = 12
                search_regions.append([region_l, region_r])
            self.log.debug("Search regions: " + str(search_regions))

            ##
            # STAGE 2 - FIND PEAK CANDIDATE INTERVALS
            ##
            # Finds regions above threshold were peak candidates may be found
            ##
            for region_left, region_right in search_regions:

                # (re)set some bookkeeping variables, need these later
                self.current_region_left = region_left
                self.current_region_right = region_right
                self.left_extent_small_s2_search_limit = self.current_region_left
                self.last_s1_rightmost_index = region_left
                self.seeker_position = region_left
                self.this_s1_alert_position = 0

                # Search for threshold crossings
                while 1:

                    if self.seeker_position >= region_right:
                        self.log.debug("Seeker position ended up at %s, right region boundary is at %s!" % (
                            self.seeker_position, region_right,
                        ))
                        break

                    # Find the left_boundary of the next peak candidate interval
                    if signal[self.seeker_position] > settings['threshold']:
                        """
                        We're already above threshold, this is the start of a new peak
                            Note on XerawDP matching:
                            If we're looking for s1s, the seeker position can count as starting an s1.
                            For S2s, Xerawdp checks for an upwards threshold CROSSING, so you would think it can't...
                            ... however, I believe this is only an issue after a peak that fails tests,
                            in this case Xerawdp sets "above-threshold" to false manually.
                        """
                        left_boundary = self.seeker_position
                    else:
                        # We're not currently above threshold, so find the next threshold crossing, if it exists
                        left_boundary = find_next_crossing(signal, threshold=settings['threshold'],
                                                           start=self.seeker_position, stop=region_right)

                    # Was there another candidate interval?
                    if left_boundary == region_right:
                        break

                    # Keep a record of the s1 alert positions: needed for isolation tests
                    if peak_type == 's1':
                        self.this_s1_alert_position = left_boundary

                    # Determine the peak candidate interval boundaries
                    if peak_type == 's1':
                        # Determine the maximum in the next 60 samples that follows
                        max_pos = int(np.argmax(signal[left_boundary:min(left_boundary + 60, region_right)]))
                        max_idx = left_boundary + max_pos
                        # Set the peak candidate interval based on the max position
                        # NB this overwrites left_boundary!

                        left_boundary = max(max_idx - 10 - 2, 10)
                        right_boundary = max_idx + 60
                        # Xerawdp doesn't do proper bounds checking:
                        #   10 should be min(region_right, "")
                        #   max_idx + 60 should be max(region_right, "")
                        # I'll do only a check to prevent crashes:
                        if right_boundary > len(signal) - 1:
                            right_boundary = len(signal) - 1
                        # Checking left boundary <0 is unnecessary, Xerawdp starts S1 search at 12.

                    else:
                        # For S2s, we simply find where we drop below threshold again
                        right_boundary = find_next_crossing(signal, threshold=settings['threshold'],
                                                            start=left_boundary, stop=region_right)
                        max_idx = None  # This will be determined later

                    # Set the seeker position for the next peak search, if needed
                    if peak_type == 'large_s2':
                        # Start looking for new intervals after this one (right_boundary itself is below threshold)
                        # In the rare case that peaks are found beyond which extent beyond the interval
                        #  it is their responsibility to update the seeker position
                        self.seeker_position = right_boundary
                    # For S1s and small s2s, the next seeker position will be set in find_peaks_in,
                    # based on the right extent of the peak found in the interval.

                    # Does th peak candidate interval actually end? (the tail of a big S2 sometimes doesn't)
                    if right_boundary == region_right:
                        # No, it didn't.
                        if peak_type == 's2':
                            # Xerawdp should only 'see' an S2 when it drops below threshold, (really? test this!)
                            # however, s2s at the end of the waveform will always end due to the Xerawdp convolution bug
                            if right_boundary == event.length() - 1:
                                pass
                            else:
                                self.log.debug("%s starting at %s didn't end at region boundary %s" % (
                                    peak_type, left_boundary, right_boundary
                                ))
                                break

                    # Prepare for stage 3

                    # Hack for xerawdp matching: small s2 has wrong signal from here on
                    if peak_type == 'small_s2':
                        signal_for_later_stages = event.get_sum_waveform('filtered_for_large_s2').samples
                    else:
                        signal_for_later_stages = signal

                    # Hand over stage 3to a function: this is needed because Xerawdp recurses for large s2s
                    self.find_peaks_in(event, peak_type, signal_for_later_stages, settings,
                                       left_boundary, right_boundary,
                                       toplevel=True, max_idx=max_idx)

                    # For large s2, update the dynamic threshold
                    # TODO: don't modify settings, so they don't have to be reloaded all the time!
                    if peak_type == 'large_s2':
                        potential_new_threshold = 0.001 * self.highest_s2_height_ever
                        if potential_new_threshold > settings['threshold']:
                            self.log.debug("Dynamic threshold raised from %s to %s" % (
                                settings['threshold'], potential_new_threshold))
                            settings['threshold'] = potential_new_threshold

        return event

    def find_peaks_in(self, event, peak_type, signal, settings,
                      left_boundary, right_boundary, toplevel=False, max_idx=None):

        ##
        # STAGE 3 - FIND PEAK EXTENTS
        ##
        self.log.debug("%s candidate interval %s - %s" % (peak_type, left_boundary, right_boundary))

        # Check if the interval is large enough to contain a peak
        # We must start with this for large s2s, as it breaks the execution right here if it fails
        failed_interval_test = False
        if not settings['min_base_interval_length'] <= \
                right_boundary - left_boundary <= settings['max_base_interval_length']:
            self.log.debug("    Interval failed width test %s <= %s <= %s" % (
                settings['min_base_interval_length'],
                right_boundary - left_boundary,
                settings['max_base_interval_length']
            ))
            if peak_type == 'large_s2':
                return
            else:
                # We can't return yet, we need the right extent of the never-to-be-accepted peak
                # for the new self.seeker_position for small_s2 and s1...
                failed_interval_test = True

        # Find the maximum index and height (value at maximum)
        if max_idx is None:
            max_idx = left_boundary + np.argmax(signal[left_boundary:right_boundary + 1])  # Remember python indexing
        height = signal[max_idx]

        # Check for ghost peaks
        # Can happen for small S2s, as Xerawdp uses different waveform for peakfinding than extent computation
        # I don't see how Xerawdp can prevent seriously madhat behavior without checking this...
        if height == 0:
            # ABORT ABORT ABORT
            self.log.debug("Ghost peak (0 height) at %s, cannot compute extent!" % max_idx)
            # Have to set seeker position... Guess:
            self.seeker_position = right_boundary + 1
            return

        # How is the point to stop looking for the peak's extent related to the interval we're searching in?
        # TODO: add more comments here
        if peak_type == 'large_s2':
            left_extent_search_limit = self.nearest_s2_boundary(event, max_idx, left_boundary, 'left')
            right_extent_search_limit = self.nearest_s2_boundary(event, max_idx, right_boundary, 'right')
        elif peak_type == 'small_s2':
            left_extent_search_limit = self.left_extent_small_s2_search_limit
            right_extent_search_limit = self.current_region_right
        else:       # peak_type == 's1':
            # The interval has already been tailored for us in stage 2
            left_extent_search_limit = left_boundary
            right_extent_search_limit = right_boundary

        # Find the peak extent
        # Xerawdp arcana: the S1 extent will be extended by 2 in both directions when it is stored,
        # but the value we find here is used for tests etc.
        (left, right) = interval_until_threshold(
            signal,
            start=max_idx,
            left_threshold=settings['left_boundary_to_height_ratio'] * height,
            right_threshold=settings['right_boundary_to_height_ratio'] * height,
            left_limit=left_extent_search_limit,
            right_limit=right_extent_search_limit,
            stop_if_start_exceeded=settings['stop_if_start_exceeded'],
            min_crossing_length=settings['min_crossing_length'],
            activate_xerawdp_hacks_for=peak_type,   # :-(
        )
        self.log.debug("%s peak candidate: %s-%s-%s" % (peak_type, left, max_idx, right))

        # For s1 and small s2s, we now finally know where to start searching for the next peak candidate interval.
        # (for large s2s we knew it in stage 2 already)
        # Set the seeker position based on the right extent of the peak found
        # This is done even if a peak is rejected, so we do it before testing.
        if peak_type != 'large_s2':
            self.seeker_position = right + 1

        ##
        # STAGE 4 - PEAK ACCEPTANCE TESTS
        ##
        # Does a variety of tests to see if we accept the peak, or throwing it away silently...
        # I'll leave the judgement of this approach to the conscience of the reader.
        ##

        # Finally, we can (and should) return if we failed the interval test already above.
        if failed_interval_test:
            return

        # Check for pathological peaks; Xerawdp doesn't do this, I'm going to because it prevents crashes
        # Hmm, that probably means you're wrong about other things, right?
        if left >= right:
            self.log.warning(
                "%s at %s-%s-%s has invalid extent -- not testing or appending it! Go debug the processor!" % (
                    peak_type, left, max_idx, right
                ))
            return
        if left < 0:
            self.log.warning("%s at %s-%s-%s starts before 0, making it start at 0 instead." % (
                peak_type, left, max_idx, right
            ))
            left = 0
        if right > len(signal) - 1:
            self.log.warning("%s at %s-%s-%s ends after waveform end, making it end at waveform end instead." % (
                peak_type, left, max_idx, right
            ))
            right = len(signal) - 1

        # Store some quantities for large s2
        if peak_type == 'large_s2':
            # Update highest s2 height ever for dynamic threshold determination.
            # This has to get done AFTER the interval test, but before the peak has a chance to fail any other test
            # TODO: Can we also update the threshold itself here, or does that give problems?
            self.highest_s2_height_ever = max(self.highest_s2_height_ever, height)
            # We need this for tests later
            # Specifically, to see if child intervals contain peaks which can skip some of the large_s2
            if toplevel:
                self.last_toplevel_max_val = height

        # Apply various tests to the peaks
        if peak_type == 'large_s2':

            # For large S2 from top-level interval: do the isolation check on the top interval
            if toplevel:
                if not self.isolation_test(signal, max_idx,
                                           right_edge_of_left_test_region=min(left_boundary, left) - 1,
                                           test_length_left=settings['test_around_interval'],
                                           left_edge_of_right_test_region=max(right_boundary, right) + 1,
                                           test_length_right=settings['test_around_interval'],
                                           before_avg_max_ratio=settings['around_interval_to_height_ratio_max'],
                                           after_avg_max_ratio=settings['around_interval_to_height_ratio_max'],
                                           can_fail_one=True):  # NB Different from all the others!!!! (also s1?)
                    self.log.debug("    Toplevel interval failed isolation test")
                    return

            # The peak FWHM is tested only for lage s2s #TODO: HMMZ a duplicate fwhm computation...
            # Hack for Xerawdp matching: FWHM test now ends at boundaries, but this introduces missed peaks!
            fwhm = extent_until_threshold(signal[left:right + 1], start=max_idx - left, threshold=height / 2)
            if not settings['min_length'] <= fwhm <= settings['max_length']:
                self.log.debug("    Failed width test")
                return

            # An additional, different, isolation test is applied to every individual peak < 0.05 the toplevel peak
            if height < 0.05 * self.last_toplevel_max_val:
                if not self.isolation_test(signal, max_idx,
                                           right_edge_of_left_test_region=left - 1,
                                           test_length_left=settings['test_around_peak'],
                                           left_edge_of_right_test_region=right + 1,
                                           test_length_right=settings['test_around_peak'],
                                           before_avg_max_ratio=settings['around_peak_to_height_ratio_max'],
                                           after_avg_max_ratio=settings['around_peak_to_height_ratio_max']):
                    self.log.debug("    Failed isolation test")
                    return

        elif peak_type == 'small_s2':

            # Test for aspect ratio of the UNFILTERED WAVEFORM, probably to avoid misidentifying s1s as small s2s
            unfiltered_signal = event.get_sum_waveform('uS2').samples
            height_for_aspect_ratio_test = unfiltered_signal[left + int(np.argmax(unfiltered_signal[left:right + 1]))]
            aspect_ratio_threshold = settings['aspect_ratio_threshold']
            peak_width = right - left
            aspect_ratio = height_for_aspect_ratio_test / peak_width
            self.log.debug("    Aspect ratio test: %s <? threshold %s" % (aspect_ratio, aspect_ratio_threshold))
            if aspect_ratio > aspect_ratio_threshold:
                self.log.debug('    Failed!')
                return

            # For small s2's the isolation test is slightly different
            # This is insane, and probably a bug, but I swear it's in Xerawdp
            left_isolation_test_window = min(settings['test_around'], right_boundary - left_extent_search_limit)
            right_isolation_test_window = min(settings['test_around'], right_extent_search_limit - right_boundary)
            self.log.debug("    Isolation test windows used: left %s, right %s" % (
                left_isolation_test_window, right_isolation_test_window
            ))
            if not self.isolation_test(signal, max_idx,
                                       right_edge_of_left_test_region=left - 1,
                                       left_edge_of_right_test_region=right + 1,
                                       test_length_left=left_isolation_test_window,
                                       test_length_right=right_isolation_test_window,
                                       before_avg_max_ratio=settings['around_to_height_ratio_max'],
                                       after_avg_max_ratio=settings['around_to_height_ratio_max']):
                self.log.debug("    Failed the isolation test.")
                return

        elif peak_type == 's1':
            # Test for non-isolated peaks
            # TODO: dynamic window size if several s1s close together, check with Xerawdp now that free_regions
            left_isolation_test_window = min(settings['test_before'],
                                             self.this_s1_alert_position - self.last_s1_rightmost_index)
            right_isolation_test_window = min(settings['test_after'],
                                              self.current_region_right - self.this_s1_alert_position)
            if not self.isolation_test(signal, max_idx,
                                       right_edge_of_left_test_region=left - 1,
                                       test_length_left=left_isolation_test_window,
                                       left_edge_of_right_test_region=right + 1,
                                       test_length_right=right_isolation_test_window,
                                       before_avg_max_ratio=settings['before_avg_max_ratio'],
                                       after_avg_max_ratio=settings['after_avg_max_ratio']):
                return

            # Test for nearby negative excursions #Xerawdp bug: no check if is actually negative..
            # Todo: check for off-by one errors...
            # Need +1 for python indexing? Xerawdp doesn't do 1-correction here?
            negex = min(signal[
                max_idx - left_isolation_test_window:
                max_idx + right_isolation_test_window
            ])
            if not height > 3 * abs(negex):
                self.log.debug('    Failed negative excursion test')
                return

            # Test for too wide s1s, using the FWQM of a filtered waveform made specially for this test
            # In reality this test is less strict than it seems:
            #  - the filtered waveform is mangled by the convolution bug, so the widths come out lower
            #  - Xerawdp limits the width search quite strictly, making it come out lower than the real FWQM
            filtered_wave = event.get_sum_waveform('filtered_for_s1_width_test').samples
            # not right+1, Xerawdp doesn't include it either
            max_in_filtered = left + int(np.argmax(filtered_wave[left:right]))
            if filtered_wave[max_in_filtered] <= 0:
                # = 0 Happens due to Xerawdp's convolution bug, Xerawdp's width will return 0, passes test
                # <0 happens for very short&shallow s1s, it breaks my implementation of extent_until_threshold,
                #    since it searches for crossings, which it never finds). Xerawdp it should return a filtered
                #    width of 0 since it is already below threshold, so it always passes the test.
                pass
            else:
                filtered_width = extent_until_threshold(filtered_wave[left_boundary:right_boundary + 1],
                                                        start=max_in_filtered - left_boundary,
                                                        threshold=0.25 * filtered_wave[max_in_filtered])
                if filtered_width > settings['max_filtered_width']:
                    self.log.debug('    Filtered width %s larger than %s' % (
                        filtered_width, settings['max_filtered_width']))
                    return

        # Update values for later isolation tests
        # This gets done only if the current peak passes all tests!
        # It has to get done AFTER the isolation test of the current peak of course.
        # TODO: maybe we can keep track of these in a more intelligent way? They're both boundaries of last peak...
        if peak_type == 'small_s2':
            # For small s2s, we need to store the right peak boundary as the left boundary for later isolation tests.
            self.left_extent_small_s2_search_limit = right
        elif peak_type == 's1':
            # For s1, update the last_s1_right_boundary, also used for later isolation tests
            self.last_s1_rightmost_index = right

        ##
        # STAGE 5 - STORING THE PEAK
        ##
        # Does what it says on the tin.
        ##

        # If we're still here, append the peak found
        if peak_type == 's1':
            # Arcane Xerawdp stuff, probably to compensate for after crossing timer stuff?
            left -= 2
            right += 2
        self.log.debug("! Appending %s (%s-%s-%s) to peaks" % (peak_type, left, max_idx, right))
        event.peaks.append(datastructure.Peak({
            'type':             's1' if peak_type == 's1' else 's2',    # Don't want large_s2 and small_s2 here
            'area':             np.sum(signal[left:right]),   # Waste of time, only gets used for sorting s2s...
            'index_of_maximum': max_idx,
            'height':           height,
            'left':             left,
            'right':            right,
            'detector':         'tpc',
        }))

        # Recursion for large s2s
        if peak_type == 'large_s2':
            # For large s2s, ACCEPTED peaks beyond the interval get to set the next seeker position
            if right > self.seeker_position:
                self.seeker_position = right + 1
            # Recurse on remaining intervals right, then left
            # Ordering is different from Xerawdp because it uses a stack, we use recursion
            self.find_peaks_in(event, peak_type, signal, settings,
                               left_boundary=left_boundary, right_boundary=left)  # Not left-1? Who is off by one?
            self.find_peaks_in(event, peak_type, signal, settings,
                               left_boundary=right, right_boundary=right_boundary)  # Not right+1? Who is off by one?

    # Helper methods used only in the peakfinding
    def nearest_s2_boundary(self, event, peak_position, edge_position, direction):
        """Finds the nearest s2 boundary according to arcane Xerawdp rules"""
        boundaries = []
        if direction == 'left':
            # Will take the max of 0, edge_position-100, and any s2 right boundaries before peak
            boundaries = [0, edge_position - 100]
            for p in event.S2s():
                if p.right <= peak_position:
                    # The = case actually happens!
                    boundaries.append(p.right)
            return max(boundaries)
        elif direction == 'right':
            # Will take the min of event_duration-1, edge_position+100, and any s2 left boundaries after peak
            boundaries = [event.length() - 1, edge_position + 100]
            for p in event.S2s():
                if peak_position <= p.left:
                    # The = case actually happens!
                    boundaries.append(p.left)  # TODO: Was this a Xerawdp bug? or should you now redo the matching?
            return min(boundaries)
        raise RuntimeError("Direction %s isn't left or right" % direction)

    def isolation_test(
        self,
        signal, max_idx,
        right_edge_of_left_test_region, test_length_left,
        left_edge_of_right_test_region, test_length_right,
        before_avg_max_ratio,
        after_avg_max_ratio,
        can_fail_one=False,
    ):
        """Does XerawDP's arcane isolation test. Returns if test is passed or not.
        NB: edge = first index IN region to test!
        TODO: Regions seem to come out empty sometimes, which triggers a numpy runtiome warning (only once though..)

        If you compare the isolation test intervals with xerawdp, it looks like ours are too large.
        In reality, Xerawdp's math is off, and we are testing the same intervals.
        """
        # Xerawdp off-by-one error:
        right_edge_of_left_test_region -= (1 if test_length_left % 2 == 0 else 0)
        # Xerawdp reports 1 tighter boundaries in the debug output on both sides than it actually tests;
        # I could do the same to let the log files also match, but that's taking the whole matching idea too far..
        self.log.debug("        Running isolation test: RofLeft %s, LofRight %s, LengthLeft %s, LengthRight %s" % (
            right_edge_of_left_test_region, left_edge_of_right_test_region, test_length_left, test_length_right
        ))
        height = signal[max_idx]
        # For empty test regions, Xerawdp gives 0.0/0 = some kind of Nan for average, ensuring test always fails
        if test_length_left == 0:
            self.log.debug("        Empty left test region! Auto-failing before-avg test.")
            failed_pre = True
            pre_avg = 'not computed'
        else:
            # +1s are to compensate for python's indexing conventions...
            pre_avg = np.mean(signal[
                right_edge_of_left_test_region - (test_length_left - 1):
                right_edge_of_left_test_region + 1
            ])
            failed_pre = pre_avg > height * before_avg_max_ratio
        if test_length_right == 0:
            self.log.debug("        Empty right test region! Auto-failing after-avg test.")
            failed_post = True
            post_avg = 'not computed'
        else:
            post_avg = np.mean(signal[
                left_edge_of_right_test_region:
                left_edge_of_right_test_region + test_length_right
            ])
            failed_post = post_avg > height * after_avg_max_ratio
        self.log.debug("        Fail test if before avg %s > %s and/or above avg %s > %s" % (
            pre_avg, height * before_avg_max_ratio, post_avg, height * after_avg_max_ratio
        ))
        if can_fail_one:
            failed = failed_pre and failed_post
        else:
            failed = failed_pre or failed_post
        if failed:
            self.log.debug("        Failed!")
            return False
        else:
            self.log.debug("        Passed!")
            return True


# Sort and prune a peak list - needed in peakfinder and Basics
def sort_and_prune_by(peak_list, key=lambda x: x, keep_number=float('inf'), reverse=False):
    peak_list.sort(key=key, reverse=reverse)
    if len(peak_list) > keep_number:
        return peak_list[:keep_number]
    else:
        return peak_list


class ComputePeakProperties(plugin.TransformPlugin):

    """Compute various derived quantities of each peak (full width half maximum, etc.)


    """

    def transform_event(self, event):
        """Only computes area for Xerawdp matching at the moment
        """

        # Compute relevant peak quantities for each pmt's peak: height, FWHM, FWTM, area, ..
        for peak in event.peaks:
            peak.area_per_channel = np.zeros(len(event.channel_waveforms), dtype='float64')
            for channel, wave_data in enumerate(event.channel_waveforms):
                if channel not in self.config['channels_in_detector']['tpc'] or \
                   peak.type == 's1' and channel in self.config['channels_excluded_for_s1']:
                    continue
                # No +1, Xerawdp forgets the right edge also:
                peak.area_per_channel[channel] = np.sum(wave_data[peak.left:peak.right])
            # Exclude negative areas
            peak.area = sum([area for _, area in enumerate(peak.area_per_channel) if area > 0])
            """
            The coincidence level is actually computed twice in Xerawdp: once before and once after gain correction
            The coincidence computed before gain correction is used for sorting
                (and saved in the peak data object, but not used (I think/hope so?))
            The coincidence computed  after gain correction is stored in the root file
            For now we'll compute only the one before gain correction, so we can implement the sorting
            TODO: implement the other one too!
            """
            if peak.type == 's1':
                contributing_pmts = []
                for channel, area in enumerate(peak.area_per_channel):
                    if channel not in self.config['channels_in_detector']['tpc'] \
                            or channel in self.config['channels_excluded_for_s1']:
                        continue
                    if self.config['gains'][channel] == 0:
                        continue
                    if area > self.config['coincidence_threshold'] * (2 * 10 ** 6 / self.config['gains'][channel]):
                        contributing_pmts.append(channel)
                peak.does_channel_contribute = np.array(
                    [ch in contributing_pmts for ch in range(self.config['n_channels'])],
                    dtype=np.bool)
            else:
                # Hack to ensure S2s won't get pruned:
                peak.does_channel_contribute = np.ones(self.config['n_channels'], dtype=np.bool)

            peak.n_contributing_channels = len(peak.contributing_channels)
            peak.does_channel_have_noise = np.zeros(self.config['n_channels'], dtype=np.bool)

        # Prune excess S1s
        event.peaks = sort_and_prune_by(
            event.peaks,
            key=lambda x: (x.n_contributing_channels, x.area),
            keep_number=(len(event.peaks) - len(event.S1s()) + 32 if self.config['prune_peaks'] else float('inf')),
            reverse=True
        )

        return event


# Helper functions for peakfinding
# Can't yet put them in the peakfinding class, because extent_until_threshold is used by computequantities also...

log = logging.getLogger('PeakFinding_find_next_crossing')


def find_next_crossing(signal, threshold,
                       start=0, direction='right', min_length=1,
                       stop=None, stop_if_start_exceeded=False, activate_xerawdp_hacks_for=None):
    """Returns first index in signal crossing threshold, searching from start in direction
    A 'crossing' is defined as a point where:
        start_sample < threshold < this_sample OR start_sample > threshold > this_sample

    Arguments:
    signal            --  List of signal samples to search in
    threshold         --  Threshold defining crossings
    start             --  Index to start search from: defaults to 0
    direction         --  Direction to search in: 'right' (default) or 'left'
    min_length        --  Crossing only counts if stays above/below threshold for min_length
                          Default: 1, i.e, a single sample on other side of threshold counts as a crossing
                          The first index where a crossing happens is still returned.
    stop_at           --  Stops search when this index is reached, THEN RETURNS THIS INDEX!!!
    #Hack for Xerawdp matching:
    stop_if_start_exceeded      -- If true and a value HIGHER than start is encountered, stop immediately
    activate_xerawdp_hacks_for  -- Activates several hacks, for instance, slope checking for large s2

    This is a pretty crucial function for several DSP routines; as such, it does extensive checking for
    pathological cases. Please be very careful in making changes to this function, their effects could
    be felt in unexpected ways in many places.

    TODO: add lots of tests!!!
    TODO: Allow specification of where start_sample should be (below or above treshold),
          only use to throw error if it is not true? Or see start as starting crossing?
           -> If latter, can outsource to new function: find_next_crossing_above & below
              (can be two lines or so, just check start)
    TODO: Allow user to specify that finding a crossing is mandatory / return None when none found?

    """

    # Set stop to last index along given direction in signal
    if stop is None:
        stop = 0 if direction == 'left' else len(signal) - 1

    # Check for errors in arguments
    # TEMP HACK, these should become errors... or at least stern warnings!
    warninghead = ("Strange error during Xerawdp-imitation peakfinding: find_next_crossing was asked to search for " +
                   "a threshold crossing for >= %s samples in the %s direction starting from sample %s and stopping " +
                   "at sample %s in a signal %s samples in size.\n") % (min_length, direction, start, stop, len(signal))
    if stop < 0:
        log.warning(warninghead + "A negative stop point makes no sense, setting it to 0 instead.")
        stop = 0
    if stop > len(signal) - 1:
        stop = len(signal) - 1
        log.warning(warninghead + "A stop point beyond the signal extent makes no sense, setting %s instead." % stop)
    if start < 0:
        log.warning(warninghead + "A negative start point makes no sense, setting %s instead." % stop)
        start = 0
    if start > len(signal) - 1:
        start = len(signal) - 1
        log.warning(warninghead + "A start point beyond the signal extent makes no sense, setting %s instead." % start)
    # These are already errors
    if direction not in ('left', 'right'):
        raise ValueError(warninghead + "Direction %s is not left or right" % direction)
    if not 1 <= min_length:
        raise ValueError(warninghead + "Crossing for %s samples makes no sense, should be >= 1." % min_length)
    if (direction == 'left' and start < stop) or (direction == 'right' and stop < start):
        raise RuntimeError("Start and stop points are not in right order for search direction.")

    # Check for pathological cases which can arise, not serious enough to throw an exception
    # Can't raise a warning from here, as I don't have self.log...
    if stop == start:
        return stop
    if signal[start] == threshold:
        log.warning((warninghead + 'However, the signal is EXACTLY equal to the threshold at the start position, so ' +
                     "the very idea of 'crossing' is ambiguous. Will return %s as the crossing position") % stop)
        return stop
    if not min_length <= abs(start - stop):
        # This is probably ok, can happen, remove warning later on
        log.warning((warninghead + "The search region is too small for a threshold crossing of this length " +
                     "to occur in it. Will return %s as the crossing position. See issue #68.") % stop)
        return stop

    # Do the search
    i = start
    after_crossing_timer = 0
    start_sample = signal[start]
    while 1:
        this_sample = signal[i]
        if i == stop:
            # stop_at reached, have to return something right now
            if activate_xerawdp_hacks_for == 'large_s2':
                # We need to index of the minimum before & including this point instead...
                if direction == 'left':
                    return i + np.argmin(signal[i:start + 1])
                else:    # direction=='right'
                    return start + np.argmin(signal[start:i + 1])
            elif activate_xerawdp_hacks_for == 's1':
                # Xerawdp keeps going, but always increments after_crossing_timer, so we know what it'll give
                # This is a completely arcane hack due to several weird interactions of boundary cases
                if not this_sample < threshold:
                    # The counter gets reset on this sample
                    after_crossing_timer = 0
                else:
                    # This sample increments the counter
                    after_crossing_timer += 1
                return stop + (-1 + after_crossing_timer if direction == 'left' else 1 - after_crossing_timer)
            elif activate_xerawdp_hacks_for == 'small_s2':  # small_s2
                return stop + (-1 if direction == 'left' else 1)
            else:
                return stop     # Sane case, doesn't happen in Xerawdp I think
        if stop_if_start_exceeded and this_sample > start_sample:
            # print("Emergency stop of search at %s: start value exceeded" % i)
            return i
        if start_sample < threshold < this_sample or start_sample > threshold > this_sample:
            # We're on the other side of the threshold that at the start!
            after_crossing_timer += 1
            if after_crossing_timer == min_length:
                return i + (min_length - 1 if direction == 'left' else 1 - min_length)
        else:
            # We're back to the old side of threshold again
            after_crossing_timer = 0

        # Dirty hack for Xerawdp matching
        if activate_xerawdp_hacks_for == 'large_s2':
            # Check also for slope inversions
            if this_sample > 7.801887059:  # '0.125 V'
                # We need to check for slope inversions. How bad is it allowed to be?
                if this_sample < 39.00943529:  # '0.625 V'
                    log_slope_threshold = 0.02  # Xerawdp says '0.02 V/bin', but it is a log slope threshold...
                else:
                    log_slope_threshold = 0.005  # Idem '0.005 V/bin'

                # Todo: check if enough samples exist to compute slope..
                try:
                    # Calculate the slope at this point using XeRawDP's '9-tap derivative kernel'
                    log_slope = np.sum(signal[i - 4:i + 5] * np.array([
                        -0.003059, -0.035187, -0.118739, -0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059
                    ])) / this_sample
                    # print("Slope is %s, threshold is %s" % (slope, slope_threshold))
                    # Left slopes of peaks are positive, so a negative slope indicates inversion
                    # If slope inversions are seen, return index of the minimum before this.
                    if direction == 'left' and log_slope < - log_slope_threshold:
                        # print("    ! Inversion found on rising (left) slope at %s:
                        # log_slope %s < - %s. Started from %s" %  (i, log_slope, log_slope_threshold, start))
                        return i + np.argmin(signal[i:start + 1])
                    elif direction == 'right' and log_slope > log_slope_threshold:
                        # print("    ! Inversion found on falling (right) slope at %s:
                        # log_slope %s > %s. started from %s" % (i, log_slope, log_slope_threshold, start))
                        return start + np.argmin(signal[start:i + 1])
                except ValueError:
                    log.warning("The slope inversion test crashed, tell Jelle he should not be lazy and check if " +
                                "there are enough samples to use the derivative kernel.")
        # Increment the search position in the right direction
        i += -1 if direction == 'left' else 1


def interval_until_threshold(signal, start,
                             left_threshold, right_threshold=None, left_limit=0, right_limit=None,
                             min_crossing_length=1, stop_if_start_exceeded=False, activate_xerawdp_hacks_for=None
                             ):
    """Returns (l,r) indices of largest interval including start on same side of threshold
    ... .bla... bla
    """
    if right_threshold is None:
        right_threshold = left_threshold
    if right_limit is None:
        right_limit = len(signal) - 1
    l_cross = find_next_crossing(signal, left_threshold,  start=start, stop=left_limit,
                                 direction='left',  min_length=min_crossing_length,
                                 stop_if_start_exceeded=stop_if_start_exceeded,
                                 activate_xerawdp_hacks_for=activate_xerawdp_hacks_for)
    r_cross = find_next_crossing(signal, right_threshold, start=start, stop=right_limit,
                                 direction='right', min_length=min_crossing_length,
                                 stop_if_start_exceeded=stop_if_start_exceeded,
                                 activate_xerawdp_hacks_for=activate_xerawdp_hacks_for)
    # BADNESS, but needed for Xerawdp matching: This ins't interval_until_threshold, but 1 more on each side!
    return (l_cross, r_cross)


def extent_until_threshold(signal, start, threshold):
    a = interval_until_threshold(signal, start, threshold)
    return a[1] - a[0]


class Filtering(plugin.TransformPlugin):

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def startup(self):

        if self.config['simulate_Xerawdp_convolution_bug']:
            self.log.warning('Xerawdp convolution bug simulation is enabled: S2 peak finding will be less accurate.')

        for f in self.config['filters']:
            self.log.debug("Applying filter %s" % f['name'])

            if 'impulse_response' in f:
                # Time domain filter: convolution with an impulse response
                ir = np.array(f['impulse_response'])

                # Check if the impulse response is sane

                if abs(1 - np.sum(ir)) > 0.0001:
                    self.log.warning("Filter %s has non-normalized impulse response: %s != 1."
                                     "Normalizing for you..." % (f['name'], np.sum(ir)))
                    f['impulse_response'] = ir / np.sum(ir)

                if len(ir) % 2 == 0:
                    self.log.warning("Filter %s has an even-length impulse response!" % f['name'])

                if not np.all(ir - ir[::-1] == 0):
                    self.log.warning("Filter %s has an asymmetric impulse response!" % f['name'])

            else:
                # Frequency bandpass filter
                # Implementation from http://wiki.scipy.org/Cookbook/ButterworthBandpass
                sampling_rate = 1 / self.config['sample_duration']

                # Plot the frequency response for a few different orders.
                # fs = sampling_rate
                # import matplotlib.pyplot as plt
                # plt.figure(1)
                # plt.clf()
                # for order in range(6):
                #     b, a = self.butter_bandpass(
                #          lowcut=f['low_freq_bound'],
                #          highcut=f['high_freq_bound'],
                #          fs=sampling_rate,
                #          order=order
                #     )
                #     w, h = freqz(b, a, worN=2000)
                #     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
                #
                # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
                #          '--', label='sqrt(0.5)')
                # plt.xlabel('Frequency (GHz)')
                # plt.ylabel('Gain')
                # plt.grid(True)
                # plt.legend(loc='best')

                f['filter_parameters'] = self.butter_bandpass(
                    lowcut=f['low_freq_bound'],
                    highcut=f['high_freq_bound'],
                    fs=sampling_rate,
                    order=1    # At high orders the frequency response seems to go bananas..?
                )

    def transform_event(self, event):
        for f in self.config['filters']:

            input_w = event.get_sum_waveform(f['source'])
            signal = input_w.samples
            if 'impulse_response' in f:
                output = np.convolve(signal, f['impulse_response'], 'same')
            else:
                output = filtfilt(f['filter_parameters'][0], f['filter_parameters'][1], signal)

            if self.config['simulate_Xerawdp_convolution_bug'] and 'impulse_response' in f:
                ##
                # This dirty code hack implements the Xerawdp convolution bug
                # DO NOT USE except for Xerawdp matching!
                ##
                # TODO: could be done more straightforwardly now that we've stored pulses properly
                filter_length = len(f['impulse_response'])

                # Determine the pulse boundaries
                y = np.abs(np.sign(signal))
                pbs = np.concatenate((np.where(np.roll(y, 1) - y == -1)[0],
                                      np.where(np.roll(y, -1) - y == -1)[0]))

                # Check if these are real pulse boundaries: at least three samples before or after must be zero
                real_pbs = []
                for q in pbs:
                    if q < 3 or q > len(signal) - 4:
                        continue  # So these tests don't fail
                    if signal[q - 1] == signal[q - 2] == signal[q - 3] == 0 or \
                       signal[q + 1] == signal[q + 2] == signal[q + 3] == 0:
                        real_pbs.append(q)

                # Mutilate the waveform
                # First mutilate the edges, which are always pulse boundaries
                output[:int(filter_length / 2)] = np.zeros(int(filter_length / 2))
                output[len(output) - int(filter_length / 2):] = np.zeros(int(filter_length / 2))
                # Mutilate waveform around pulse boundaries
                for pb in real_pbs:
                    try:
                        lefti = max(0, pb - int(filter_length / 2))
                        righti = min(len(signal) - 1, pb + int(filter_length / 2))
                        output[lefti:righti] = np.zeros(righti - lefti)
                    except Exception as e:
                        self.log.warning("Error during waveform mutilation: " + str(e) + ". So what...")

            event.sum_waveforms.append(datastructure.SumWaveform(
                name=f['name'],
                samples=output,
                channel_list=input_w.channel_list,
                detector=input_w.detector,
            ))

        return event
