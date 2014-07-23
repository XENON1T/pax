import logging
import inspect
import collections
from pprint import pprint

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

class Event(object):
    def __init__(self, raw={}):
        self.log = logging.getLogger('Event')
        self._raw = raw

    @property
    def raw(self, i_know_what_i_am_doing = False):
        """Do not use: raw internals."""
        if not i_know_what_i_am_doing:
            self.log.warning("Using RAW event data.")
        return self._raw

    def event_window(self):
        """The real-world time of the start and end of event

        64-bit number since 1/1/1970
        """
        return (0.0, 0.0)

    def _get_peaks(self, inputs, sort):
        """Fetch S1 or S2 peaks from our data structure
        """
        # Sort key is used on the flattened peak
        # Sort order is whether or not Python sort is 'reversed'
        sort_key, sort_order = sort

        peaks = {}

        for peak in self.raw['peaks']:
            # 'input' refers to which filtered or summed waveform the peak was
            # computed on. 
            if peak['input'] in inputs and peaks['rejected'] == False:
                # Flatten the peak so we can use our sort key
                peak_key = flatten(peak)[sort_key]

                pprint(flatten(peak))

                # Save save into dictionary
                peaks[peak_key] = S2(peak)

        # Return just a list, but sorted according to our sort key and order
        return [peaks[i] for i in sorted(peaks, reverse=sort_order)]

    def S2s(self, sort=('top_and_bottom.area', True)):
        """List of S2 (ionization) signals as Peak objects"""
        return self._get_peaks(('filtered_for_large_s2',
                               'filtered_for_small_s2'),
                               sort)

    def S1s(self, sort=('area', True)):
        """List of S1 (scintillation) signals as Peak objects"""
        return self._get_peaks(('uncorrected_sum_waveform_for_s1'), sort)

    def pmt_waveform(self, pmt):
        """The individual waveform for a specific PMT"""
        return self.raw.channel_waveforms[pmt]

    def summed_waveform(self, name='top_and_bottom'):
        """Waveform summed over many PMTs"""
        if 'filtered' in name:
            raise ValueError('Do not get filtered waveforms with summed waveform; use filtered_waveform: %s' % name)
        elif name not in self.raw['processed_waveform']:
            raise ValueError("Summed waveform %s does not exist")

        return self.raw['processed_waveform'][name]

    def filtered_waveform(self):
        """Filtered waveform summed over many PMTs"""
        return self.raw['processed_waveform']['filtered_for_large_s2']

    def dump(self):
        pprint.pprint(self.raw)


class Peak(object):
    def __init__(self, peak_dict):
        self.peak_dict = {}

    def _get_var(pmts, key):
        key = '%s.%s' % (pmts, key)
        flattened_peak = flatten(self.peak_dict)
        if key not in flattened_peak:
            raise ValueError('%s does not exist in peak' % key)
        
        return flattened_peak[key]

    def area(key='top_and_bottom'):
        return self._get_var(key, 'area')

    def width_fwhm(key='top_and_bottom'):
        return self._get_var(key, 'fwhm')

    def height(key='top_and_bottom'):
        return self._get_var(key, 'height')

    def time_in_waveform(key='top_and_bottom'):
        return self._get_var(key, 'position_of_max_in_waveform')

class S1(Peak):
    pass

class S2(Peak):
    pass

def explain(class_name):
    x = inspect.getmembers(class_name,
                           predicate=inspect.isdatadescriptor)

    for a, b in x:
        if a.startswith('_'):
            continue
        print(a, b.__doc__)

if __name__ == '__main__':
    explain(Peak)
    explain(Event)


