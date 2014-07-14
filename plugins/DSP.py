"""Digital signal processing"""
import numpy as np

from pax import plugin, units, dsputils

__author__ = 'tunnell'

##
## Utils
##
    
def baseline_mean_stdev(waveform, sample_size=46):
    """ returns (baseline, baseline_stdev), calculated on the first sample_size samples of waveform """
    baseline_sample = waveform[:sample_size]
    return (
        np.mean(sorted(baseline_sample)[
            int(0.4*len(baseline_sample)):int(0.6*len(baseline_sample))
        ]),  #Ensures peaks in baseline sample don't skew computed baseline
        np.std(baseline_sample)                     #... but do count towards baseline_stdev!
    )
    #Don't want to just take the median as V-resolution is finite
    #Don't want the mean either: this is not robust against large fluctuations (eg peaks in sample)

def extent_until_treshold(signal, start, treshold):
    a = interval_until_treshold(signal, start, treshold)
    return a[1]-a[0]

def interval_until_treshold(signal, start, treshold):
    return (
        find_first_below(signal, start, treshold, 'left'),
        find_first_below(signal, start, treshold, 'right'),
    )
    
def find_first_below(signal, start, treshold, direction, min_length_below=1):
    #TODO: test for off-by-one errors
    counter = 0
    if direction=='right':
        for i,x in enumerate(signal[start:]):
            if x<treshold: return start+i
    elif direction=='left':
        i = start
        while 1:
            if signal[i]<treshold:
                counter += 1
                if counter == min_length_below:
                    return i    #or i-min_length_below ??? #TODO
            else:
                counter = 0
            if direction=='right':
                i += 1
            elif direction=='left':
                i -= 1
            else:
                raise(Exception, "You nuts? %s isn't a direction!" % direction)
                
def all_same_length(items): return all(len(x) == len(items[0]) for x in items) 
    
def rcosfilter(filter_length, rolloff, cutoff_freq, sampling_freq=1):
    """
    Returns a nd(float)-array describing a raised cosine (RC) filter (FIR) impulse response. Arguments:
        - filter_length:    filter length in samples
        - rolloff:          roll-off factor
        - cutoff_freq:      cutoff frequency = 1/(2*symbol period)
        - sampling_freq:    sampling rate (in same units as cutoff_freq)
    """
    Ts = 1/(2*cutoff_freq)
    h_rc = np.zeros(filter_length, dtype=float)
        
    for x in np.arange(filter_length):
        t = (x-filter_length/2)/float(sampling_freq)
        phase = np.pi*t/Ts
        if t == 0.0:
            h_rc[x] = 1.0
        elif rolloff != 0 and abs(t) == Ts/(2*rolloff):
            h_rc[x] = (np.pi/4)*(np.sin(phase)/phase)
        else:
            h_rc[x] = (np.sin(phase)/phase)* (np.cos(phase*rolloff)/(1-(((2*rolloff*t)/Ts)*((2*rolloff*t)/Ts))))
    
    return h_rc/h_rc.sum()
    
    
##
##  Classes
##
    
    
    
    
class JoinAndConvertWaveforms(plugin.TransformPlugin):

    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)
        #Maybe we should get dt and dV from input format if possible?
        self.conversion_factor = config[
            'digitizer_V_resolution'] * config['digitizer_t_resolution']
        self.conversion_factor /= config['gain']
        self.conversion_factor /= config['digitizer_resistor']
        self.conversion_factor /= config['digitizer_amplification']
        self.conversion_factor /= units.electron_charge
        
    def TransformEvent(self, event):
        if 'channel_waveforms' in event:
            #Data is not ZLE, we only need to baseline correct & convert
            for channel, wave in event['channel_waveforms'].items():
                baseline, _ = dsputils.baseline_mean_stdev(wave) 
                event['channel_waveforms'][channel] -= baseline
                event['channel_waveforms'][channel] *= -1 * self.conversion_factor
        elif 'channel_occurences' in event:
            #Data is ZLE, we need to build the waves from occurences
            event['channel_waveforms'] = {}
            for channel, waveform_occurences in event['channel_occurences'].items():
                #Determine an average baseline for this channel, using all the occurences
                baseline    = np.mean([dsputils.baseline_mean_stdev(wave_occurence)[0] 
                                       for _, wave_occurence in waveform_occurences])
                wave = np.ones(event['length']) * baseline
                #Put wave occurences in the right positions
                for starting_position, wave_occurence in waveform_occurences:
                    wave[starting_position:starting_position+len(wave_occurence)] = wave_occurence
                event['channel_waveforms'][channel] = -1 * (wave - baseline) * self.conversion_factor
            del event['channel_occurences']
        return event

class ComputeSumWaveform(plugin.TransformPlugin):

    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)

        # TODO (tunnell): These keys should come from configuration?
        self.channel_groups = {'top': config['top'],
                               'bottom': config['bottom'],
                               'veto': config['veto'],
                               'summed': config['top'] + config['bottom']}

    def TransformEvent(self, event):
        sum_waveforms = {}
        # Compute summed waveforms
        for group, members in self.channel_groups.items():
            sum_waveforms[group] = sum([wave for name, wave in event['channel_waveforms'].items() if name in members])
            if type(sum_waveforms[group]) != type(np.array([])):
                # None of the group members have a waveform in this event,
                # delete this group's waveform
                sum_waveforms.pop(group)
                continue

        event['sum_waveforms'] = sum_waveforms
        return event


class GenericFilter(plugin.TransformPlugin):
        
    def apply_filter_by_convolution(self, signal, normalized_impulse_response):
        """
        Filters signal using specified impulse-response, using convolution
        """
        return np.convolve(signal, normalized_impulse_response, 'same')
        
    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)

    def TransformEvent(self, event):
        if 'filtered_waveforms' not in event:
            event['filtered_waveforms'] = {}
        event['filtered_waveforms'][self.output_name] = self.apply_filter_by_convolution(event['sum_waveforms'][self.input_name], self.filter_ir)
        return event

class LargeS2Filter(GenericFilter):
    def __init__(self, config):
        GenericFilter.__init__(self, config)
        self.filter_ir = rcosfilter(31, 0.2, 3*units.MHz*config['digitizer_t_resolution'])
        self.output_name = 'filtered_for_large_s2'
        self.input_name = 'summed'
        
class SmallS2Filter(GenericFilter):
    def __init__(self, config):
        GenericFilter.__init__(self, config)
        self.filter_ir = [0, 0.103, 0.371, 0.691, 0.933, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.933, 0.691, 0.371, 0.103, 0]
        self.output_name = 'filtered_for_small_s2'
        self.input_name = 'summed'

        

class PeakFinder_X100style(plugin.TransformPlugin):

    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)


    def X100_style(self,
                   signal,
                   treshold=10,
                   boundary_to_height_ratio=0.1,
                   min_length=1,
                   max_length=float('inf'),
                   test_before=1,
                   before_to_height_ratio_max=float('inf'),
                   test_after=1,
                   after_to_height_ratio_max=float('inf'),
                   **kwargs):
        """
        First finds pre-peaks: intervals above treshold for which
            * min_length <= length <= max_length
            * mean of test_before samples before interval must be less than before_to_height_ratio_max times the maximum value in the interval
            * vice versa for after
        Then looks for peaks in the pre-peaks: boundary whenever it first drops below boundary_to_max_ratio*height
        """

        # Find any prepeaks
        prepeaks = []
        new = {'prepeak_left': 0}
        previous = float("-inf")
        for i, x in enumerate(signal):
            if x > treshold and previous < treshold:
                new['prepeak_left'] = i
            elif x < treshold and previous > treshold:
                new['prepeak_right'] = i
                prepeaks.append(new)
                new = {'prepeak_left': i}  # can't new={}, in case this is start of new peak already... wait, that can't happen right?
            previous = x
        # TODO: Now at end of waveform: any unfinished peaks left

        # Filter out prepeaks that don't meet conditions
        valid_prepeaks = []
        for b in prepeaks:
            if not min_length <= b['prepeak_right'] - b['prepeak_left'] <= max_length:
                continue
            b['index_of_max_in_prepeak'] = np.argmax(signal[b['prepeak_left']: b[
                'prepeak_right'] + 1])  # Remember python indexing... Though probably right boundary isn't ever the max!
            b['index_of_max_in_waveform'] = b[
                'index_of_max_in_prepeak'] + b['prepeak_left']
            b['height'] = signal[b['index_of_max_in_waveform']]
            b['before_mean'] = np.mean(
                signal[max(0, b['prepeak_left'] - test_before): b['prepeak_left']])
            b['after_mean'] = np.mean(
                signal[b['prepeak_right']: min(len(signal), b['prepeak_right'] + test_after)])
            if b['before_mean'] > before_to_height_ratio_max * b['height'] or b[
                    'after_mean'] > after_to_height_ratio_max * b['height']:
                continue
            valid_prepeaks.append(b)

        # Find peaks in the prepeaks
        # TODO: handle presence of multiple peaks in base, that's why I make a
        # new array already now
        peaks = []
        for p in valid_prepeaks:
            # TODO: stop find_first_below search if we reach boundary of an
            # earlier peak? hmmzz need to pass more args to this. Or not
            # needed?
            (p['left'], p['right']) = interval_until_treshold(signal,
                                                                   start=p[
                                                                       'index_of_max_in_waveform'],
                                                                   treshold=boundary_to_height_ratio * p['height'])
            p['peak_type'] = self.output_peak_type
            peaks.append(p)

        return peaks

    def TransformEvent(self, event):
        if 'peaks' not in event:
            event['peaks'] = []
        event['peaks'] += self.X100_style(self.get_input_waveform(event), **self.peakfinder_settings)
        return event
        
class LargeS2Peakfinder(PeakFinder_X100style):
    def __init__(self, config):
        PeakFinder_X100style.__init__(self, config)
        dt = self.config['digitizer_t_resolution']
        self.get_input_waveform = lambda event : event['filtered_waveforms']['filtered_for_large_s2']
        self.output_peak_type = 'large_s2'
        self.peakfinder_settings = {
            'treshold'    : 0.62451,
            'left_boundary_to_height_ratio' : 0.005,
            'right_boundary_to_height_ratio': 0.002,
            'min_length'  : int(0.6 *units.us/dt),
            'test_before' : int(0.21*units.us/dt),
            'test_after'  : int(0.21*units.us/dt),
            #Different settings for top level interval and intermediate interval
            #Take this into account for peak splitter!
            'before_to_height_ratio_max'  : 0.05,
            'after_to_height_ratio_max'   : 0.05
        }
        
class SmallS2Peakfinder(PeakFinder_X100style):
    def __init__(self, config):
        PeakFinder_X100style.__init__(self, config)
        dt = self.config['digitizer_t_resolution']
        self.get_input_waveform = lambda event : event['filtered_waveforms']['filtered_for_small_s2']
        self.output_peak_type = 'large_s2'
        self.peakfinder_settings = {
            'treshold'    : 0.062451,
            'left_boundary_to_height_ratio' : 0.01,
            'right_boundary_to_height_ratio': 0.01,
            'min_length'  : int(0.4 *units.us/dt),
            'test_before' : int(0.1*units.us/dt),
            'test_after'  : int(0.1*units.us/dt),
            'before_to_height_ratio_max'  : 0.05,
            'after_to_height_ratio_max'   : 0.05
        }
        
class S1Peakfinder(PeakFinder_X100style):
    def __init__(self, config):
        PeakFinder_X100style.__init__(self, config)
        dt = self.config['digitizer_t_resolution']
        self.get_input_waveform = lambda event : event['sum_waveforms']['summed']
        self.output_peak_type = 's1'
        self.peakfinder_settings = {
            'treshold'    : 0.1872453,
            'left_boundary_to_height_ratio'  : 0.005,
            'right_boundary_to_height_ratio' : 0.005,
            'min_samples_below_boundary'     : 3,
            'max_length'  : int(0.6 *units.us/dt),
            'test_before' : int(0.5*units.us/dt),
            'test_after'  : int(0.1*units.us/dt),
            'before_to_height_ratio_max'  : 0.01,
            'after_to_height_ratio_max'   : 0.04
        }
        
class VetoS1Peakfinder(S1Peakfinder):
    def __init__(self, config):
        S1Peakfinder.__init__(self, config)
        self.get_input_waveform = lambda event : event['sum_waveforms']['veto']
        self.output_peak_type = 'veto_s1'
    


class ComputeQuantities(plugin.TransformPlugin):

    def TransformEvent(self, event):
        """For every filtered waveform, find peaks
        """

        #Compute relevant peak quantities for each pmt's peak: height, FWHM, FWTM, area, ..
        #Todo: maybe clean up this data structure? This way it was good for csv..
        peaks = event['peaks']
        for i, p in enumerate(peaks):
            for channel, data in event['sum_waveforms'].items():
                #Todo: use python's handy arcane naming/assignment convention to beautify this code
                peak_wave = data[p['left']:p['right']+1]    #Remember silly python indexing
                peaks[i][channel] = {}
                maxpos = peaks[i][channel]['position_of_max_in_peak']= np.argmax(peak_wave)
                max = peaks[i][channel]['height']                    = peak_wave[maxpos]
                peaks[i][channel]['position_of_max_in_waveform']     = p['left'] + maxpos
                peaks[i][channel]['area']                            = np.sum(peak_wave)
                if channel == 'summed':
                    #Expensive stuff, only do for summed waveform, maybe later for top&bottom as well?
                    samples_to_ns = self.config['digitizer_t_resolution']/units.ns
                    peaks[i][channel]['fwhm'] = extent_until_treshold(peak_wave, start=maxpos, treshold=max/2)  *samples_to_ns
                    peaks[i][channel]['fwqm'] = extent_until_treshold(peak_wave, start=maxpos, treshold=max/4)  *samples_to_ns
                    peaks[i][channel]['fwtm'] = extent_until_treshold(peak_wave, start=maxpos, treshold=max/10) *samples_to_ns
            if 'top' in peaks[i] and 'bottom' in peaks[i]:
                peaks[i]['asymmetry'] = (peaks[i]['top']['area']-peaks[i]['bottom']['area'])/(peaks[i]['top']['area']+peaks[i]['bottom']['area'])

        return event
