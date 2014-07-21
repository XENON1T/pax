"""Digital signal processing

This starts from the raw channel occurrences and spits out (after running the
 many plugins) a list of peaks.

"""
import numpy as np
from copy import copy

from pax import plugin, units
from pax.dsputils import baseline_mean_stdev, find_next_crossing, extent_until_threshold, interval_until_threshold


class BuildUncorrectedSumWaveformForXerawdpMatching(plugin.TransformPlugin):
    def startup(self):
        config = self.config

        # Conversion factor from converting from ADC counts -> pe/bin
        self.conversion_factor = config['digitizer_t_resolution'] * config['digitizer_voltage_range'] / (
            2 ** (config['digitizer_bits']) * config['pmt_circuit_load_resistor'] *
            config['external_amplification'] * units.electron_charge
        )

    def transform_event(self, event):
        wave = np.zeros(event['length'])
        for channel, waveform_occurrences in event['channel_occurrences'].items():
            if channel >178 or channel in [1, 2, 145, 148, 157, 171, 177]:
                continue
            for starting_position, wave_occurrence in waveform_occurrences:
                baseline, _ = baseline_mean_stdev(wave_occurrence)
                wave[starting_position:starting_position + len(wave_occurrence)] = \
                     np.add(-1 * (wave_occurrence - baseline) * self.conversion_factor / (2*10**6),
                            wave[starting_position:starting_position + len(wave_occurrence)]
                     )
        if not 'processed_waveforms' in event:
            event['processed_waveforms'] = {}
        event['processed_waveforms']['uncorrected_sum_waveform_for_xerawdp_matching'] = wave
        return event



class JoinAndConvertWaveforms(plugin.TransformPlugin):

    """Take channel_occurrences, builds channel_waveforms

    Between occurrence waveforms (i.e. pulses...), zeroes are added.
    The waveforms returned will be converted to pe/ns and baseline-corrected.
    If a channel is absent from channel_occurrences, it wil be absent from channel_waveforms.

    """

    def startup(self):
        # Short hand
        c = self.config
        self.gains = c['gains']

        # Conversion factor from converting from ADC counts -> pe/bin
        self.conversion_factor = c['digitizer_t_resolution'] * c['digitizer_voltage_range'] / (
            2 ** (c['digitizer_bits']) * c['pmt_circuit_load_resistor']
            * c['external_amplification'] * units.electron_charge
        )

    def transform_event(self, event):
        # Check if voltage range same as reported by input plugin
        # TODO: do the same for dt
        if 'metadata' in event and 'voltage_range' in event['metadata']:
            if event['metadata']['voltage_range'] != self.config['digitizer_voltage_range']:
                raise RuntimeError(
                    'Voltage range from event metadata (%s) is different from ini file setting (%s)!'
                    % (event['metadata']['voltage_range'], self.config['digitizer_voltage_range'])
                )

        # Check for input plugin misbehaviour / running this plugin at the wrong time
        if not ('channel_occurrences' in event and 'length' in event):
            raise RuntimeError(
                "Event contains %s, should contain at least channel_occurrences and length !"
                % str(event.keys())
            )

        # Build the channel waveforms from occurrences
        event['channel_waveforms'] = {}
        for channel, waveform_occurrences in event['channel_occurrences'].items():

            # Check that gain known
            if channel not in self.gains:
                self.log.warning('Gain for channel %s is not specified! Skipping channel.' % channel)
                continue

            # Deal with unknown gains
            if self.gains[channel] == 0:
                if channel in event['channel_occurrences']:
                    self.log.warning('Gain for channel %s is 0, but is in waveform.' % channel)
                    continue
                else:
                    # Just a dead channel, no biggie
                    continue

            # Assemble the waveform pulse by pulse, starting from an all-zeroes waveform
            wave = np.zeros(event['length'])
            for starting_position, wave_occurrence in waveform_occurrences:
                # Determine an average baseline for this occurrence
                """
                This is NOT THE WAY TO DO IT - we should at least average over all occurrences
                Better yet, take a mean of median 10% or so
                Better yet, do this for several events, keep a running mean
                However, this is how Xerawdp does it... (see Rawdata.cpp, getPulses)
                # TODO: Check for baseline fluctuations in event, warn if too much
                # How much baseline will we have in 1T? Only few samples?
                See also comment in baseline_mean_stdev
                """
                baseline, _ = baseline_mean_stdev(wave_occurrence)
                # Put wave occurrences in the correct positions
                wave[starting_position:starting_position + len(wave_occurrence)] = wave_occurrence - baseline

            # Flip the waveform up, convert it to pe/ns, and store it in the event data structure
            event['channel_waveforms'][channel] = -1 * wave * self.conversion_factor / self.gains[channel]


        # Delete the channel_occurrences from the event structure, we don't need it anymore
        del event['channel_occurrences']

        return event


class ComputeSumWaveform(plugin.TransformPlugin):

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
        self.channel_groups['top_and_bottom'] = self.channel_groups['top'] | self.channel_groups['bottom']
        # TEMP for XerawDP matching: Don't have to compute peak finding waveform yet, done in JoinAndConvertWaveforms

    def transform_event(self, event):
        if not 'processed_waveforms' in event:
            event['processed_waveforms'] = {}

        # Compute summed waveforms
        for group, members in self.channel_groups.items():
            event['processed_waveforms'][group] = np.zeros(event['length'])
            for channel in members:
                if channel in event['channel_waveforms']:
                    event['processed_waveforms'][group] += event['channel_waveforms'][channel]

        return event


class GenericFilter(plugin.TransformPlugin):

    """Generic filter base class

    Do not instantiate. Instead, subclass: subclass has to set
        self.filter_ir  --  filter impulse response (normalized, i.e. sum to 1)
        self.input_name      --  label of waveform in processed_waveforms to filter
        self.output_name     --  label where filtered waveform is stored in processed_waveforms

    TODO: Add some check the name of the class and throw exception if base class
          is instantiated.  use self.name.
    TODO: check if ir normalization;
    """
    # Always takes input from a wave in processed_waveforms

    def startup(self):
        self.filter_ir = None
        self.output_name = None
        self.input_name = None

    def transform_event(self, event):
        # Check if we have all necessary information
        if self.filter_ir is None or self.output_name is None or self.input_name is None:
            raise RuntimeError('Filter subclass did not provide required parameters')
        if round(sum(self.filter_ir), 4) != 1.:
            raise RuntimeError('Impulse response sums to %s, should be 1!' % sum(self.filter_ir))

        event['processed_waveforms'][self.output_name] = np.convolve(
            event['processed_waveforms'][self.input_name],
            self.filter_ir,
            'same'
        )
        return event


class LargeS2Filter(GenericFilter):

    """Docstring  Low-pass filter using raised cosine filter

    TODO: put constants into ini?
    """

    def startup(self):
        GenericFilter.startup(self)

        self.filter_ir = self.rcosfilter(31, 0.2, 3 * units.MHz * self.config['digitizer_t_resolution'])
        self.output_name = 'filtered_for_large_s2'
        self.input_name = 'uncorrected_sum_waveform_for_xerawdp_matching'

    @staticmethod
    def rcosfilter(filter_length, rolloff, cutoff_freq, sampling_freq=1):
        """
        Returns a nd(float)-array describing a raised cosine (RC) filter (FIR) impulse response. Arguments:
            - filter_length:    filter length in samples
            - rolloff:          roll-off factor
            - cutoff_freq:      cutoff frequency = 1/(2*symbol period)
            - sampling_freq:    sampling rate (in same units as cutoff_freq)
        """
        symbol_period = 1 / (2 * cutoff_freq)
        h_rc = np.zeros(filter_length, dtype=float)

        for x in np.arange(filter_length):
            t = (x - filter_length / 2) / float(sampling_freq)
            phase = np.pi * t / symbol_period
            if t == 0.0:
                h_rc[x] = 1.0
            elif rolloff != 0 and abs(t) == symbol_period / (2 * rolloff):
                h_rc[x] = (np.pi / 4) * (np.sin(phase) / phase)
            else:
                h_rc[x] = (np.sin(phase) / phase) * (
                    np.cos(phase * rolloff) / (
                        1 - (((2 * rolloff * t) / symbol_period) * ((2 * rolloff * t) / symbol_period))
                    )
                )

        return h_rc / h_rc.sum()


class SmallS2Filter(GenericFilter):

    """

    TODO: take this opportunity to explain why there is a small s2 filter... even if it stupid.
    TODO: put constants into ini?
    """

    def startup(self):
        GenericFilter.startup(self)

        self.filter_ir = np.array([0, 0.103, 0.371, 0.691, 0.933, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.933, 0.691,
                                   0.371, 0.103, 0])
        self.filter_ir = self.filter_ir / sum(self.filter_ir)  # Normalization
        self.output_name = 'filtered_for_small_s2'
        self.input_name = 'uncorrected_sum_waveform_for_xerawdp_matching'


class FindS1_XeRawDPStyle(plugin.TransformPlugin):

    def transform_event(self, event):
        signal = event['processed_waveforms']['uncorrected_sum_waveform_for_xerawdp_matching']
        s1_alert_treshold = 0.1872453  # "3 mV"
        # TODO: set start&end positions based on regions where S2s are found, loop over intervals
        left_region_limit = 0
        seeker_position = copy(left_region_limit)
        right_region_limit = len(signal) - 1
        while 1:
            # Seek until we go above s1_alert_treshold
            potential_s1_start = find_next_crossing(signal, s1_alert_treshold,
                                                    start=seeker_position, stop=right_region_limit)
            if potential_s1_start == right_region_limit:
                # Search has reached the end of the waveform
                break

            # Determine the maximum in the 'preliminary peak window' (next 60 samples) that follows
            max_pos = np.argmax(signal[potential_s1_start:min(potential_s1_start + 60, right_region_limit)])
            max_idx = potential_s1_start + max_pos
            height = signal[max_idx]

            # Set a revised peak window based on the max position
            # Should we also check for previous S1s? or prune overlap later? (Xerawdp doesn't do either?)
            peak_window = (
                max(max_idx - 8, left_region_limit),
                min(max_idx + 60, right_region_limit)
            )

            # Find the peak boundaries
            peak_bounds = interval_until_threshold(signal, start=max_idx,
                                                   left_threshold=0.005*height, #r-trh automatically =
                                                   left_limit=peak_window[0], right_limit=peak_window[1],
                                                   min_crossing_length=3, stop_if_start_exceeded=True)

            # Don't have to test for s1 > 60: can't happen due to limits on peak window

            #Add the s1 we found
            event['peaks'].append({
                'peak_type':                's1',
                'left':                     peak_bounds[0],
                'right':                    peak_bounds[1],
                's1_peakfinding_window':    peak_window,
                's1_peakfinder_original_alert': seeker_position,
                'index_of_max_in_waveform': max_idx,
                'height':                   height,
                'input':                    'top_and_bottom',
            })

            #Continue searching after the peak
            seeker_position = copy(peak_bounds[1])


        return event


# TODO: add a self cleaning option to the ini file for tossing out middle steps?
# TODO: Veto S1 peakfinding
class PrepeakFinder(plugin.TransformPlugin):

    """Finds intervals 'above threshold' for which min_length <= length <= max_length.

    'above threshold': dropping below threshold for shorter than max_samples_below_threshold is acceptable

    This needs a lot more explaination.  Few paragrahs?

    Toss out super big or small peaks without warning.  Compute left, right, height.
    TODO: Call it peak candidates
    """

    # Doc for future version:
    """Adds candidate peak intervals to peak_candidate_intervals

    A peak candidate is an interval above threshold for which min_length <= length <= max_length.
    Dropping below threshold for shorter than max_samples_below_threshold does not count as ending the interval.
    If an interval is above treshold, but does not meet the length conditions set, it is not reported!
    peak_candidate_intervals will be a list of dicts with left, right, height.

    """

    def startup(self):
        self.settings = {  # TODO put in ini
                           'threshold': {'large_s2': 0.62451, 'small_s2': 0.062451},
                           'min_length': {'large_s2': 60, 'small_s2': 40},
                           'max_length': {'large_s2': float('inf'), 'small_s2': 200},
                           'input': {
                               'large_s2': 'filtered_for_large_s2',
                               'small_s2': 'filtered_for_small_s2'
                           }
        }

    def transform_event(self, event):
        event['prepeaks'] = []

        # Which peak types do we need to search for?
        peak_types = self.settings['threshold'].keys()
        for peak_type in peak_types:
            prepeaks = []
            # Find which settings to use for this type of peak
            settings = {}
            for settingname, settingvalue in self.settings.items():
                settings[settingname] = self.settings[settingname][peak_type]

            # Get the signal out
            signal = event['processed_waveforms'][settings['input']]

            # Find the prepeaks
            end = 0
            while 1:
                start = find_next_crossing(signal, threshold=settings['threshold'], start=end)
                # If we get the last index, that means it could not find another crossing
                if start == len(signal) - 1:
                    break
                end = find_next_crossing(signal, threshold=settings['threshold'],
                                         start=start)
                prepeaks.append({
                    'prepeak_left': start,
                    'prepeak_right': end,
                    'peak_type': peak_type,
                    'input': settings['input']
                })
                if end == len(signal) - 1:
                    print("Peak starting at %s didn't end!" % start)
                    break

            # Filter out prepeaks that don't meet width conditions, compute some quantities
            valid_prepeaks = []
            for b in prepeaks:
                if not settings['min_length'] <= b['prepeak_right'] - b['prepeak_left'] <= settings['max_length']:
                    continue
                # Remember python indexing... Though probably right boundary isn't ever the max!
                b['index_of_max_in_prepeak'] = np.argmax(signal[b['prepeak_left']: b['prepeak_right'] + 1])
                b['index_of_max_in_waveform'] = b['index_of_max_in_prepeak'] + b['prepeak_left']
                b['height'] = signal[b['index_of_max_in_waveform']]
                valid_prepeaks.append(b)
            # Store the valid prepeaks found
            event['prepeaks'] += valid_prepeaks

        return event


class FindPeaksInPrepeaks(plugin.TransformPlugin):

    """Put condition on height

    Looks for peaks in the pre-peaks:
    starts from max, walks down, stops whenever signal drops below boundary_to_max_ratio*height
    """

    def startup(self):
        self.settings = {  # TOOD: put in ini
                           'left_boundary_to_height_ratio': {'large_s2': 0.005, 'small_s2': 0.01},
                           'right_boundary_to_height_ratio': {'large_s2': 0.002, 'small_s2': 0.01},
        }

    def transform_event(self, event):
        event['peaks'] = []
        # Find peaks in the prepeaks
        # TODO: handle presence of multiple peaks in base, that's why I make a
        # new array already now
        for peak in event['prepeaks']:
            # Find which settings to use for this type of peak
            settings = {}
            for settingname, settingvalue in self.settings.items():
                settings[settingname] = self.settings[settingname][peak['peak_type']]
            signal = event['processed_waveforms'][peak['input']]
            max_idx = peak['index_of_max_in_waveform']

            # Determine overlap-free region
            # TODO: right now peaks which will be rejected also block this.
            # Need to move the pruning logic back in here... may as well merge prepeak & peak finding into s2 finding
            free_region = [0, len(signal)-1]
            for p in event['peaks']:
                if p['left'] <= max_idx <= p['right']:
                    free_region = None
                    break
                if free_region[0] < p['right'] < max_idx:
                    free_region[0] = p['right']
                if max_idx < p['left'] < free_region[1]:
                    free_region[1] = p['left']
            if free_region == None:
                continue

            #Deal with copying once we go into peak splitting
            (peak['left'], peak['right']) = interval_until_threshold(
                signal,
                start=max_idx,
                left_threshold=settings['left_boundary_to_height_ratio'] * peak['height'],
                right_threshold=settings['right_boundary_to_height_ratio'] * peak['height'],
                left_limit=free_region[0],
                right_limit=free_region[1],
            )
            event['peaks'].append(peak)
        return event

class ComputeQuantities(plugin.TransformPlugin):

    """Compute various derived quantities of each peak (full width half maximum, etc.)


    """

    def transform_event(self, event):
        """For every filtered waveform, find peaks
        """

        # Compute relevant peak quantities for each pmt's peak: height, FWHM, FWTM, area, ..
        # Todo: maybe clean up this data structure? This way it was good for csv..
        peaks = event['peaks']
        for i, p in enumerate(peaks):
            for channel, wave_data in event['processed_waveforms'].items():
                # Todo: use python's handy arcane naming/assignment convention to beautify this code
                peak_wave = wave_data[p['left']:p['right'] + 1]  # Remember silly python indexing
                peaks[i][channel] = {}
                maxpos = peaks[i][channel]['position_of_max_in_peak'] = np.argmax(peak_wave)
                maxval = peaks[i][channel]['height'] = peak_wave[maxpos]
                peaks[i][channel]['position_of_max_in_waveform'] = p['left'] + maxpos
                peaks[i][channel]['area'] = np.sum(peak_wave)
                if channel == 'top_and_bottom':
                    # Expensive stuff...
                    # Have to search the actual whole waveform, not peak_wave:
                    # TODO: Searches for a VERY VERY LONG TIME for weird peaks in afterpulse tail..
                    # Fix by computing baseline for inividual peak, or limiting search region...
                    # how does XerawDP do this?
                    samples_to_ns = self.config['digitizer_t_resolution'] / units.ns
                    peaks[i][channel]['fwhm'] = extent_until_threshold(wave_data, start=p['index_of_max_in_waveform'],
                                                                       threshold=maxval / 2) * samples_to_ns
                    peaks[i][channel]['fwtm'] = extent_until_threshold(wave_data, start=p['index_of_max_in_waveform'],
                                                                       threshold=maxval / 10) * samples_to_ns
                    # if 'top' in peaks[i] and 'bottom' in peaks[i]:
                    # peaks[i]['asymmetry'] = (peaks[i]['top']['area'] - peaks[i]['bottom']['area']) / (
                    # peaks[i]['top']['area'] + peaks[i]['bottom']['area'])

        return event


# #
# Utils: these are helper functions for the plugins
# TODO: Can we put these functions in the transform base class? (Tunnell: yes, or do multiple inheritance)
# #