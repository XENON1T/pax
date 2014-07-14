"""Digital signal processing"""
import numpy as np

from pax import plugin, units, dsputils

__author__ = 'tunnell'

def interval_until_treshold(signal, start, treshold):
    return (
        PeakFinder.find_first_below(signal, start, treshold, 'left'),
        PeakFinder.find_first_below(signal, start, treshold, 'right'),
    )

def extent_until_treshold(signal, start, treshold):
    a = interval_until_treshold(signal, start, treshold)
    return a[1]-a[0]
    
class JoinAndConvertWaveforms(plugin.TransformPlugin):

    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)
        #Maybe we should get dt and dV from input format if possible?
        self.conversion_factor = float(config['digitizer_V_resolution'])
        self.conversion_factor *= float(config['digitizer_t_resolution'])
        self.conversion_factor /= float(config['gain'])
        self.conversion_factor /= float(config['digitizer_resistor'])
        self.conversion_factor /= float(config['digitizer_amplification'])
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
        self.channel_groups = {'top': [int(x) for x in config['pmts_top'].split()],
                               'bottom': [int(x) for x in config['pmts_bottom'].split()],
                               'veto': [int(x) for x in config['pmts_veto'].split()],
                               'summed': ([int(x) for x in config['pmts_top'].split()] + [int(x) for x in config['pmts_bottom'].split()])}

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


class FilterWaveforms(plugin.TransformPlugin):

    def rcosfilter(self, filter_length, rolloff, cutoff_freq, sampling_freq=1):
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
        
    def apply_filter_by_convolution(self, signal, normalized_impulse_response):
        """
        Filters signal using specified impulse-response, using convolution
        """
        return np.convolve(signal, normalized_impulse_response, 'same')
        
    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)
        self.filter_ir = self.rcosfilter(31, 0.2, 3*units.MHz*float(config['digitizer_t_resolution']))

    def TransformEvent(self, event):
        event['filtered_waveforms'] = {}
        #Todo: iterate over filters
        #for filter in config['filters']
        # Key would be 'top', 'bottom', etc
        event['filtered_waveforms']['filtered_for_large_s2'] = self.apply_filter_by_convolution(event['sum_waveforms']['summed'], self.filter_ir)
        return event


class PeakFinder(plugin.TransformPlugin):

    @staticmethod
    def find_first_below(signal, start, below, direction):
        # TODO: test for off-by-one errors
        if direction == 'right':
            for i, x in enumerate(signal[start:]):
                if x < below:
                    return start + i
        elif direction == 'left':
            i = start
            while 1:
                if signal[i] < below:
                    return i
                if direction == 'right':
                    i += 1
                elif direction == 'left':
                    i -= 1
                else:
                    raise ValueError(
                        "You nuts? %s isn't a direction!" % direction)

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
                new = {
                    'prepeak_left': i}  # can't new={}, in case this is start of new peak already... wait, that can't happen right?
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
            peaks.append(p)

        return peaks

    def TransformEvent(self, event):
        """For every filtered waveform, find peaks
        """
        # Add substructure for many peak finders?
        event['peaks'] = self.X100_style(event['filtered_waveforms']['filtered_for_large_s2'], **self.config)
        #merge peaks from different peakfinders
        return event


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
