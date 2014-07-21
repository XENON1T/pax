"""Digital signal processing

This starts from the raw channel occurrences and spits out (after running the
 many plugins) a list of peaks.

"""
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

        self.gains = c['gains']

        # Conversion factor from converting from ADC counts -> pe/bin
        self.conversion_factor = c['digitizer_t_resolution']
        self.conversion_factor *= c['digitizer_voltage_range']
        self.conversion_factor /= (2 ** (c['digitizer_bits']))
        self.conversion_factor /= c['pmt_circuit_load_resistor']
        self.conversion_factor /= c['external_amplification']
        self.conversion_factor /= units.electron_charge

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

        # TEMP: for Xerawdp Matching
        # TODO: add in stuff from gain=0 waveforms!
        if not 'processed_waveforms' in event:
            event['processed_waveforms'] = {}
        event['processed_waveforms']['uncorrected_sum_waveform_for_xerawdp_matching'] = sum([
            event['channel_waveforms'][channel] * self.gains[channel] / (2 * 10 ** (6))
            for channel in event['channel_waveforms'].keys()
            if channel <= 178 and channel not in [1, 2, 145, 148, 157, 171, 177] and self.gains[channel] != 0
        ])
        # TEMP end

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
                           'threshold': {'s1': 0.1872453, 'large_s2': 0.62451, 'small_s2': 0.062451},
                           'min_length': {'s1': 0, 'large_s2': 60, 'small_s2': 40},
                           'max_length': {'s1': 60, 'large_s2': float('inf'), 'small_s2': 200},
                           'max_samples_below_threshold': {'s1': 2, 'large_s2': 0, 'small_s2': 0},
                           'input': {
                               's1': 'uncorrected_sum_waveform_for_xerawdp_matching',
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
                                         start=start, min_length=settings['max_samples_below_threshold'] + 1)
                if end == len(signal) - 1:
                    print("Peak starting at %s didn't end!" % start)
                prepeaks.append({
                    'prepeak_left': start,
                    'prepeak_right': end,
                    'peak_type': peak_type,
                    'input': settings['input']
                })

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
                           'left_boundary_to_height_ratio': {'s1': 0.005, 'large_s2': 0.005, 'small_s2': 0.01},
                           'right_boundary_to_height_ratio': {'s1': 0.005, 'large_s2': 0.002, 'small_s2': 0.01},
        }

    def transform_event(self, event):
        event['peaks'] = []
        # Find peaks in the prepeaks
        # TODO: handle presence of multiple peaks in base, that's why I make a
        # new array already now
        for p in event['prepeaks']:
            # Find which settings to use for this type of peak
            settings = {}
            for settingname, settingvalue in self.settings.items():
                settings[settingname] = self.settings[settingname][p['peak_type']]
            signal = event['processed_waveforms'][p['input']]

            # TODO: stop find_first_below search if we reach boundary of an
            # earlier peak? hmmzz need to pass more args to this. Or not
            # needed?
            # Deal with copying once we go into peak splitting
            (p['left'], p['right']) = interval_until_threshold(
                signal,
                start=p['index_of_max_in_waveform'],
                left_threshold=settings['left_boundary_to_height_ratio'] * p['height'],
                right_threshold=settings['right_boundary_to_height_ratio'] * p['height'],
            )
            event['peaks'].append(p)
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
                    # Expensive stuff, only do for summed waveform, maybe later for top&bottom as well?
                    # Have to search the actual whole waveform, not peak_wave:
                    # TODO: Searches for a VERY VERY LONG TIME for weird peaks in afterpulse tail..
                    # Fix by computing baseline for inividual peak, or limiting search region...
                    # how does XerawDP do this?
                    samples_to_ns = self.config['digitizer_t_resolution'] / units.ns
                    peaks[i][channel]['fwhm'] = extent_until_threshold(wave_data, start=p['index_of_max_in_waveform'],
                                                                       threshold=maxval / 2) * samples_to_ns
                    peaks[i][channel]['fwqm'] = extent_until_threshold(wave_data, start=p['index_of_max_in_waveform'],
                                                                       threshold=maxval / 4) * samples_to_ns
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

def baseline_mean_stdev(waveform, sample_size=46):
    """ Returns (baseline, baseline_stdev) calculated on the first sample_size samples of waveform """
    """
    This is how XerawDP does it... but this may be better:
    return (
        np.mean(sorted(baseline_sample)[
                int(0.4 * len(baseline_sample)):int(0.6 * len(baseline_sample))
                ]),  # Ensures peaks in baseline sample don't skew computed baseline
        np.std(baseline_sample)  # ... but do count towards baseline_stdev!
    )
    Don't want to just take the median as V-resolution is finite
    Don't want the mean either: this is not robust against large fluctuations (eg peaks in sample)
    See also comment in JoinAndConvertWaveforms
    """

    baseline_sample = waveform[:sample_size]
    return np.mean(baseline_sample), np.std(baseline_sample)


def find_next_crossing(signal, threshold, start=0, direction='right', min_length=1, stop=None):
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
    if not 0 <= stop <= len(signal) - 1:
        raise ValueError("Invalid crossing search stop point: %s (signal has %s samples)" % (stop, len(signal)))
    if not 0 <= start <= len(signal) - 1:
        raise ValueError("Invalid crossing search start point: %s (signal has %s samples)" % (start, len(signal)))
    if direction not in ('left', 'right'):
        raise ValueError("Direction %s is not left or right" % direction)
    if (direction == 'left' and start < stop) or (direction == 'right' and stop < start):
        raise ValueError("Search region (start: %s, stop: %s, direction: %s) has negative length!" % (
            start, stop, direction
        ))

    # Check for pathological cases not serious enough to throw an exception
    # Can't raise a warning from here, as I don't have self.log...
    if stop == start:
        return stop
    if signal[start] == threshold:
        print("Threshold %s equals value in start position %s: crossing will never happen!" % (
            threshold, start
        ))
        return stop
    if not 1 <= min_length <= abs(start - stop):
        # This is probably ok, can happen, remove warning later on
        print("Minimum crossing length %s will never happen in a region %s samples in size!" % (
            min_length, abs(start - stop)
        ))
        return stop

    # Do the search
    i = start
    after_crossing_timer = 0
    start_sample = signal[start]
    while 1:
        if i == stop:
            # stop_at reached, have to result something
            return stop
        this_sample = signal[i]
        if start_sample < threshold < this_sample or start_sample > threshold > this_sample:
            # We're on the other side of the threshold that at the start!
            after_crossing_timer += 1
            if after_crossing_timer == min_length:
                return i + (min_length - 1 if direction == 'left' else 1 - min_length)  #
        else:
            # We're back to the old side of threshold again
            after_crossing_timer = 0
        i += -1 if direction == 'left' else 1


def interval_until_threshold(signal, start, left_threshold, right_threshold=None):
    if right_threshold is None:
        right_threshold = left_threshold
    return (
        find_next_crossing(signal, left_threshold, start=start, direction='left'),
        find_next_crossing(signal, right_threshold, start=start, direction='right'),
    )


def extent_until_threshold(signal, start, threshold):
    a = interval_until_threshold(signal, start, threshold)
    return a[1] - a[0]
