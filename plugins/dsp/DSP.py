"""Digital signal processing

This starts from the raw channel occurences and spits out (after running the
 many plugins) a list of peaks.

 TODO: general comment, use line breaks!
 TODO: My main major comment is make the code look like code in other parts of the project
       otherwise it is very hard to read.  COnsult the google style guide on comments, line length
       doc strings, and other things. and white space
       And list comprehensions!
"""
import numpy as np

from pax import plugin, units

__author__ = 'tunnell'

# #
# Utils: these are helper functions for the plugins
# TODO: Can we put these functions in the transform base class?
##


def baseline_mean_stdev(waveform, sample_size=46):
    """ returns (baseline, baseline_stdev), calculated on the first sample_size samples of waveform """
    baseline_sample = waveform[:sample_size]
    ##TEMP for xerawdp matching
    return (np.mean(baseline_sample), np.std(baseline_sample))
    #This is better:
    return (
        np.mean(sorted(baseline_sample)[
                int(0.4 * len(baseline_sample)):int(0.6 * len(baseline_sample))
                ]),  # Ensures peaks in baseline sample don't skew computed baseline
        np.std(baseline_sample)  # ... but do count towards baseline_stdev!
    )
    # Don't want to just take the median as V-resolution is finite
    # Don't want the mean either: this is not robust against large fluctuations (eg peaks in sample)


def extent_until_treshold(signal, start, treshold):
    a = interval_until_treshold(signal, start, treshold)
    return a[1] - a[0]


def interval_until_treshold(signal, start, left_treshold, right_treshold=None):
    if right_treshold == None:
        right_treshold = left_treshold
    return (
        find_first_below(signal, start, left_treshold,  'left'),
        find_first_below(signal, start, right_treshold, 'right'),
    )


def find_first_below(signal, start, treshold, direction, min_length_below=1):
    # TODO: test for off-by-one errors
    counter = 0
    i = start
    while 0 < i < len(signal)-1:
        if signal[i] < treshold:
            counter += 1
            if counter == min_length_below:
                return i  # or i-min_length_below ??? #TODO
        else:
            counter = 0
        if direction == 'right':
            i += 1
        elif direction == 'left':
            i -= 1
        else:
            raise (Exception, "You nuts? %s isn't a direction!" % direction)
    #If we're here, we've reached a boundary of the waveform!
    return i


def all_same_length(items):
    return all(len(x) == len(items[0]) for x in items)


def rcosfilter(filter_length, rolloff, cutoff_freq, sampling_freq=1):
    """
    Returns a nd(float)-array describing a raised cosine (RC) filter (FIR) impulse response. Arguments:
        - filter_length:    filter length in samples
        - rolloff:          roll-off factor
        - cutoff_freq:      cutoff frequency = 1/(2*symbol period)
        - sampling_freq:    sampling rate (in same units as cutoff_freq)
    """
    Ts = 1 / (2 * cutoff_freq)
    h_rc = np.zeros(filter_length, dtype=float)

    for x in np.arange(filter_length):
        t = (x - filter_length / 2) / float(sampling_freq)
        phase = np.pi * t / Ts
        if t == 0.0:
            h_rc[x] = 1.0
        elif rolloff != 0 and abs(t) == Ts / (2 * rolloff):
            h_rc[x] = (np.pi / 4) * (np.sin(phase) / phase)
        else:
            h_rc[x] = (np.sin(phase) / phase) * (
                np.cos(phase * rolloff) / (1 - (((2 * rolloff * t) / Ts) * ((2 * rolloff * t) / Ts))))

    return h_rc / h_rc.sum()


##
# Classes: here are the actual plugins
##


class JoinAndConvertWaveforms(plugin.TransformPlugin):
    """Take separate occurences and make it into a big channel waveforms.

    This will add zeros between pulses.  We mainly do this do deal with zero
    length encoding by padding zeros.  We also do the conversion to
    photoelectrons.  Baselines are computed here.

    TODO: this is a bad class name... is there a clearer one?
    """
    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)
        # Maybe we should get dt and dV from input format if possible?
        self.config = config
        self.gains = config['gains']
        self.conversion_factor = config['digitizer_V_resolution'] * config['digitizer_t_resolution']
        self.conversion_factor /= config['digitizer_resistor']
        self.conversion_factor /= config['digitizer_amplification']
        self.conversion_factor /= units.electron_charge

    def transform_event(self, event):
        # TODO: Check dead channel map at this stage?
        # TODO: use median for baseline?
        # TODO: Warn if baseline fluctates within event?  Compute running mean?
        # TODO: Use first samples only.  How much baseline will we have in 1T?
        #       only a few samples?
        assert 'channel_occurences' in event  # TODO: Raise ValueError or RuntimeError instead...

        #Build the channel waveforms from occurences
        event['channel_waveforms'] = {}
        for channel, waveform_occurences in event['channel_occurences'].items():
            # Check that gain known
            if channel not in self.gains:
                print('Gain for channel %s is not specified! Skipping channel.' % channel)
                continue

            # Deal with unknown gains
            if self.gains[channel] == 0:
                if channel in event['channel_occurences']:
                    self.log.warning('Gain for channel %s is 0, but is in waveform.' % channel)
                    continue
                else:
                    #Just a dead channel, no biggie
                    continue

            # Determine an average baseline for this channel, using all the occurences
            baseline = np.mean([baseline_mean_stdev(wave_occurence)[0]
                                for _, wave_occurence in waveform_occurences])
            wave = np.ones(event['length']) * baseline

            # Put wave occurences in the correct positions
            for starting_position, wave_occurence in waveform_occurences:
                wave[starting_position:starting_position + len(wave_occurence)] = wave_occurence

            event['channel_waveforms'][channel] = -1 * (wave - baseline) * self.conversion_factor/self.gains[channel]
        ##TEMP: for Xerawdp Matching
        if not 'processed_waveforms' in event:
            event['processed_waveforms'] = {} 
        event['processed_waveforms']['sum_waveform_for_xerawdp_matching_that_has_been_gain_corrected_using_a_single_number'] = sum([
            event['channel_waveforms'][channel] * self.gains[channel]/(2*10**(6))
            for channel in event['channel_waveforms'].keys()
            if channel <178 and channel not in [1, 2, 145, 148, 157, 171, 177] and self.gains[channel] != 0
        ])

        #Delete the channel_occurences from the event structure, we don't need it anymore
        del event['channel_occurences']

        return event
        

class ComputeSumWaveform(plugin.TransformPlugin):
    """Build the sum waveforms for, e.g., top PMTs

    We don't have to worry about gain corrections so we can just add the
    waveforms from the JoinAndConvertWaveforms step.

    """
    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)

        self.channel_groups = {'top': config['pmts_top'] - config['pmts_bad'],
                               'bottom': config['pmts_bottom'],
                               'veto': config['pmts_veto']}

        self.channel_groups['top_and_bottom'] = self.channel_groups['top'] | self.channel_groups['bottom']

    def transform_event(self, event):
        if not 'processed_waveforms' in event:
            event['processed_waveforms'] = {}

        # Compute summed waveforms
        for group, members in self.channel_groups.items():
            event['processed_waveforms'][group] = sum([wave for name, wave in event['channel_waveforms'].items() if name in members])  # TODO: break this up into steps, list comprehensions is evil like this.

            # None of the group members have a waveform in this event,
            # delete this group's waveform, it will probably be [] or some other nasty thing that can cause crashes
            # TODO: use isinstance() ndarray
            if type(event['processed_waveforms'][group]) != type(np.array([])):
                event['processed_waveforms'].pop(group)
                continue

        return event


class GenericFilter(plugin.TransformPlugin):
    """Generic filter base class for all later filters

    Do not instantiate.

    TODO: Add some check the name of the class and throw exception if base class
          is instantiated.  use self.name.
    """
    #Always takes input from a wave in processed_waveforms

    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)

        # THis should be normalized cooeffients
        # TODO: check if noramlzied cooefccieint?
        self.filter_ir = None  # TODO: do same for input and output name

    def apply_filter_by_convolution(self, signal, normalized_impulse_response):
        """Filters signal using specified impulse-response, using convolution
        """
        return np.convolve(signal, normalized_impulse_response, 'same')


    def transform_event(self, event):
        event['processed_waveforms'][self.output_name] = self.apply_filter_by_convolution(
            event['processed_waveforms'][self.input_name], self.filter_ir)
        return event


class LargeS2Filter(GenericFilter):
    """Docstring  Low-pass filter using raised cosine filter

    TODO: put constants into ini?
    """
    def __init__(self, config):
        GenericFilter.__init__(self, config)
        self.filter_ir = rcosfilter(31, 0.2, 3 * units.MHz * config['digitizer_t_resolution'])
        self.output_name = 'filtered_for_large_s2'
        self.input_name = 'sum_waveform_for_xerawdp_matching_that_has_been_gain_corrected_using_a_single_number'


class SmallS2Filter(GenericFilter):
    """

    TODO: take this opportunity to explain why there is a small s2 filter... even if it stupid.
    TODO: put constants into ini?
    """

    def __init__(self, config):
        GenericFilter.__init__(self, config)
        self.filter_ir = np.array([0, 0.103, 0.371, 0.691, 0.933, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.933, 0.691,
                                  0.371, 0.103, 0])
        self.filter_ir = self.filter_ir / sum(self.filter_ir)  # Normalization
        self.output_name = 'filtered_for_small_s2'
        self.input_name = 'sum_waveform_for_xerawdp_matching_that_has_been_gain_corrected_using_a_single_number'
      


# TODO: add a self cleaning option to the ini file for tossing out middle steps?
# TODO: Veto S1s!!
class PrepeakFinder(plugin.TransformPlugin):
    """todo Write short string.

    Finds intervals 'above treshold' for which min_length <= length <= max_length.
    'above treshold': dropping below treshold for shorter than max_samples_below_treshold is acceptable

    This needs a lot more explaination.  Few paragrahs?

    Toss out super big or small peaks without warning.  Compute left, right, height.
    TODO: Call it peak candidates
    """

    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)

        self.settings = {  # TODO put in ini
            'treshold'          :   {'s1': 0.1872453, 'large_s2': 0.62451,      'small_s2': 0.062451}, 
            'min_length'        :   {'s1': 0,         'large_s2': 60,           'small_s2': 40}, 
            'max_length'        :   {'s1': 60,        'large_s2': float('inf'), 'small_s2': 200}, 
            'max_samples_below_treshold' :  {'s1': 2, 'large_s2': 0,            'small_s2': 0}, 
            'input' : {
                's1'      : 'sum_waveform_for_xerawdp_matching_that_has_been_gain_corrected_using_a_single_number',
                'large_s2': 'filtered_for_large_s2',
                'small_s2': 'filtered_for_small_s2'
            }
        }

    def transform_event(self, event):
        event['prepeaks'] = []

        #Which peak types do we need to search for?
        peak_types = self.settings['treshold'].keys()
        for peak_type in peak_types:
            # Find which settings to use for this type of peak
            settings = {}
            for settingname, settingvalue in self.settings.items():
                settings[settingname] = self.settings[settingname][peak_type]

            # Get the signal out
            signal = event['processed_waveforms'][settings['input']]

            # Find any prepeaks
            prepeaks = []
            blank_prepeak = {'prepeak_left': 0, 'input' : settings['input'], 'peak_type' : peak_type}  # TODO: don't make lines longer than 80 characters please
            thisnewpeak = blank_prepeak.copy()
            previous = float("-inf")
            below_treshold_counter = 0
            for i, x in enumerate(signal):
                if x > settings['treshold'] and previous < settings['treshold']:
                    #We have come above treshold
                    below_treshold_counter = 0
                    thisnewpeak['prepeak_left'] = i
                elif x < settings['treshold'] and previous > settings['treshold'] or below_treshold_counter != 0:
                    #We have dropped below treshold
                    below_treshold_counter += 1
                    if below_treshold_counter > settings['max_samples_below_treshold']:
                        #The peak has ended! Append it and start a new one
                        thisnewpeak['prepeak_right'] = i
                        prepeaks.append(thisnewpeak)
                        thisnewpeak = blank_prepeak.copy()
                        thisnewpeak['prepeak_left'] = i #in case this is start of new peak already... wait, that can't happen right?
                        below_treshold_counter = 0
                previous = x
            # TODO: Now at end of waveform: any unfinished peaks left??

            # Filter out prepeaks that don't meet width conditions, compute some quantities
            valid_prepeaks = []
            for b in prepeaks:
                if not settings['min_length'] <= b['prepeak_right'] - b['prepeak_left'] <= settings['max_length']:
                    continue
                b['index_of_max_in_prepeak'] = np.argmax(signal[b['prepeak_left']: b[
                    'prepeak_right'] + 1])  # Remember python indexing... Though probably right boundary isn't ever the max!
                b['index_of_max_in_waveform'] = b['index_of_max_in_prepeak'] + b['prepeak_left']
                b['height'] = signal[b['index_of_max_in_waveform']]
                valid_prepeaks.append(b)
            #Store the valid prepeaks found
            event['prepeaks'] += valid_prepeaks
    
        return event


class FindPeaksInPrepeaks(plugin.TransformPlugin):
    """Put condition on height

    Looks for peaks in the pre-peaks: starts from max, walks down, stops whenever signal drops below boundary_to_max_ratio*height
    """
    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)
        self.settings = {  # TOOD: put in ini
            'left_boundary_to_height_ratio'  :   {'s1': 0.005,'large_s2': 0.005, 'small_s2': 0.01}, 
            'right_boundary_to_height_ratio' :   {'s1': 0.005,'large_s2': 0.002, 'small_s2': 0.01}, 
        }
        
    def transform_event(self, event):
        event['peaks'] = []
        # Find peaks in the prepeaks
        # TODO: handle presence of multiple peaks in base, that's why I make a
        # new array already now
        for p in event['prepeaks']:
            #Find which settings to use for this type of peak
            settings = {}
            for settingname, settingvalue in self.settings.items():
                settings[settingname] = self.settings[settingname][p['peak_type']]
            signal = event['processed_waveforms'][p['input']]

            # TODO: stop find_first_below search if we reach boundary of an
            # earlier peak? hmmzz need to pass more args to this. Or not
            # needed?
            #Deal with copying once we go into peak splitting
            (p['left'], p['right']) = interval_until_treshold(
                signal,
                start = p['index_of_max_in_waveform'],
                left_treshold  = settings['left_boundary_to_height_ratio']  * p['height'],
                right_treshold = settings['right_boundary_to_height_ratio'] * p['height'],
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
            for channel, data in event['processed_waveforms'].items():
                # Todo: use python's handy arcane naming/assignment convention to beautify this code
                peak_wave = data[p['left']:p['right'] + 1]  # Remember silly python indexing
                peaks[i][channel] = {}
                maxpos = peaks[i][channel]['position_of_max_in_peak'] = np.argmax(peak_wave)
                max = peaks[i][channel]['height'] = peak_wave[maxpos]
                peaks[i][channel]['position_of_max_in_waveform'] = p['left'] + maxpos
                peaks[i][channel]['area'] = np.sum(peak_wave)
                if channel == 'top_and_bottom':
                    # Expensive stuff, only do for summed waveform, maybe later for top&bottom as well?
                    samples_to_ns = self.config['digitizer_t_resolution'] / units.ns
                    peaks[i][channel]['fwhm'] = extent_until_treshold(peak_wave, start=maxpos,
                                                                      treshold=max / 2) * samples_to_ns
                    peaks[i][channel]['fwqm'] = extent_until_treshold(peak_wave, start=maxpos,
                                                                      treshold=max / 4) * samples_to_ns
                    peaks[i][channel]['fwtm'] = extent_until_treshold(peak_wave, start=maxpos,
                                                                      treshold=max / 10) * samples_to_ns
            # if 'top' in peaks[i] and 'bottom' in peaks[i]:
                # peaks[i]['asymmetry'] = (peaks[i]['top']['area'] - peaks[i]['bottom']['area']) / (
                    # peaks[i]['top']['area'] + peaks[i]['bottom']['area'])

        return event
