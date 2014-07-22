"""Digital signal processing

This starts from the raw channel occurrences and spits out (after running the
 many plugins) a list of peaks.

"""
import numpy as np
from copy import copy

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
        event['processed_waveforms'] = {}
        uncorrected_sum_wave = np.zeros(event['length'])
        event['channel_waveforms']   = {}
        baseline_sample_size         = 46 #TODO: put in config
        for channel, waveform_occurrences in event['channel_occurrences'].items():
            skip_channel = False  # Temp for Xerawdp matching, refactor to continue's later

            # Check that gain known
            if channel not in self.gains:
                self.log.warning('Gain for channel %s is not specified! Skipping channel.' % channel)
                skip_channel = True

            # Deal with unknown gains
            if self.gains[channel] == 0:
                if channel in event['channel_occurrences']:
                    self.log.warning('Gain for channel %s is 0, but is in waveform.' % channel)
                skip_channel = True

            # Assemble the waveform pulse by pulse, starting from an all-zeroes waveform
            wave = np.zeros(event['length'])
            for i, (starting_position, wave_occurrence) in enumerate(waveform_occurrences):

                # Check for pulses starting right after previous ones: Xerawdp doesn't recompute baselines
                if i !=0 and starting_position == waveform_occurrences[i-1][0]+len(waveform_occurrences[i-1][1]):
                    pass #baseline will still have the right value
                else:
                    # We need to compute the baseline.
                    # First pulse is allowed be short (?), then Xerawdp computes baseline from last samples instead.
                    if False and len(wave_occurrence)<2*baseline_sample_size: # Xerawdp bug, this code is never reached???
                        if not i==0:
                            raise RuntimeError("Occurrence %s in channel %s has length %s, should be at least 2*%s!"
                                               % (i, channel, len(wave_occurrence), baseline_sample_size)
                            )
                        print("Short first pulse, computing baseline from latest samples")
                        baseline_sample = wave_occurrence[len(wave_occurrence)-baseline_sample_size:]
                    else:
                        baseline_sample = wave_occurrence[:baseline_sample_size]
                    baseline = np.mean(baseline_sample)
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

                # Temp for Xerawdp matching: add pulse to the uncorrected sum waveform if they are not excluded
                if not (channel > 178 or channel in [1, 2, 145, 148, 157, 171, 177]):
                    uncorrected_sum_wave[starting_position:starting_position + len(wave_occurrence)] = \
                        np.add(-1 * (wave_occurrence - baseline) * self.conversion_factor / (2*10**6),
                            uncorrected_sum_wave[starting_position:starting_position + len(wave_occurrence)]
                        )

                # Put wave occurrences in the correct positions
                if skip_channel: continue
                wave[starting_position:starting_position + len(wave_occurrence)] = wave_occurrence - baseline

            if skip_channel: continue
            # Flip the waveform up, convert it to pe/ns, and store it in the event data structure
            event['channel_waveforms'][channel] = -1 * wave * self.conversion_factor / self.gains[channel]

        # Temp for Xerawdp matching: store uncorrected sum waveform
        event['processed_waveforms']['uncorrected_sum_waveform_for_xerawdp_matching'] = uncorrected_sum_wave

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
        s1_alert_treshold = 0.1872452894  # "3 mV"
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
                max(max_idx - 10 -2, left_region_limit),
                min(max_idx + 60, right_region_limit)
            )

            # Find the peak boundaries
            peak_bounds = interval_until_threshold(signal, start=max_idx,
                                                   left_threshold=0.005*height, #r-trh automatically =
                                                   left_limit=peak_window[0], right_limit=peak_window[1],
                                                   min_crossing_length=3, stop_if_start_exceeded=True)

            # Next search should start after this peak - do this before testing the peak or the arcane +2
            seeker_position = copy(peak_bounds[1])

            # Test for non-isolated peaks
            # Possible off-by-one error here (and elsewhere..) due to Xerawdp's arcane average syntax
            if    np.mean(signal[max(0, peak_bounds[0] - 50 -1): peak_bounds[0]]) > 0.01 * height \
               or np.mean(signal[peak_bounds[1]+1: min(len(signal), peak_bounds[1] + 10+1)]) > 0.04 * height:
                #'peak is not isolated enough'
                continue

            # Test for nearby negative excursions #Xerawdp bug: no check if is actually negative..
            negex = min(signal[
                max(0,max_idx-50 +1) :              #Need +1 for python indexing, Xerawdp doesn't do 1-correction here
                min(len(signal)-1,max_idx + 10 +1)
            ])
            if not height > 3 * abs(negex):
                #'Nearby negative excursion of %s, height (%s) not at least %s x as large.' % (negex, maxval, factor)
                continue

            #Test for too wide s1s
            filtered_wave = event['processed_waveforms']['filtered_for_large_s2']   #I know, but that's how Xerawdp...
            max_in_filtered = peak_bounds[0] + np.argmax(filtered_wave[peak_bounds[0]:peak_bounds[1]])
            filtered_width = extent_until_threshold(filtered_wave,
                                                    start=max_in_filtered,
                                                    threshold=0.25*filtered_wave[max_in_filtered])
            if filtered_width > 50:
                #'S1 FWQM in filtered_wv is %s samples, higher than 50.' % filtered_width
                continue

            # Xerawdp weirdness
            peak_bounds = (peak_bounds[0]-2,peak_bounds[1]+2)

            # Don't have to test for s1 > 60: can't happen due to limits on peak window
            # That's nonsense btw!

            #Add the s1 we found
            event['peaks'].append({
                'peak_type':                's1',
                'left':                     peak_bounds[0],
                'right':                    peak_bounds[1],
                's1_peakfinding_window':    peak_window,
                'index_of_max_in_waveform': max_idx,
                'height':                   height,
                'input':                    'uncorrected_sum_waveform_for_xerawdp_matching',
            })




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

def find_next_crossing(signal, threshold,
                       start=0, direction='right', min_length=1,
                       stop=None, stop_if_start_exceeded=False):
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
    stop_if_start_exceeded -- If true and a value HIGHER than start is encountered, stop immediately

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
    if not 1 <= min_length:
       raise ValueError("min_length must be at least 1, %s specified." %  min_length)

    # Check for pathological cases not serious enough to throw an exception
    # Can't raise a warning from here, as I don't have self.log...
    if stop == start:
        return stop
    if signal[start] == threshold:
        print("Threshold %s equals value in start position %s: crossing will never happen!" % (
            threshold, start
        ))
        return stop
    if not min_length <= abs(start - stop):
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
            # stop_at reached, have to return something right now
            # Xerawdp keeps going, but always increments after_crossing_timer, so we know what it'll give
            return stop + after_crossing_timer if direction == 'left' else stop - after_crossing_timer
        this_sample = signal[i]
        if stop_if_start_exceeded and this_sample > start_sample:
            print("Emergency stop of search at %s: start value exceeded" % i)
            return i
        if start_sample < threshold < this_sample or start_sample > threshold > this_sample:
            # We're on the other side of the threshold that at the start!
            after_crossing_timer += 1
            if after_crossing_timer == min_length:
                return i + (min_length - 1 if direction == 'left' else 1 - min_length)  #
        else:
            # We're back to the old side of threshold again
            after_crossing_timer = 0
        i += -1 if direction == 'left' else 1


def interval_until_threshold(signal, start,
                             left_threshold, right_threshold=None, left_limit=0, right_limit=None,
                             min_crossing_length=1, stop_if_start_exceeded=False
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
                           stop_if_start_exceeded=stop_if_start_exceeded)
    r_cross = find_next_crossing(signal, right_threshold, start=start, stop=right_limit,
                           direction='right', min_length=min_crossing_length,
                           stop_if_start_exceeded=stop_if_start_exceeded)
    if l_cross != left_limit:
        l_cross += 1    #We want the last index still on ok-side!
    if r_cross != right_limit:
        r_cross -= 1    #We want the last index still on ok-side!
    return (l_cross, r_cross)


def extent_until_threshold(signal, start, threshold):
    a = interval_until_threshold(signal, start, threshold)
    return a[1] - a[0]