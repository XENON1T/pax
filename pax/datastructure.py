"""Data structure for pax

This is meant to be a fixed data structure that people can use to access
physically meaningful variables.  For example, S2.

NOTE: This class is stable within major releases.  Do not change any variable
names of functionality between major releases.  You may add variables in minor
releases.  Patch releases cannot modify this.
"""

import logging
import inspect
from pprint import pprint

import collections


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
    def raw(self, i_know_what_i_am_doing=False):
        """Do not use: raw internals."""
        if not i_know_what_i_am_doing:
            self.log.warning("Using RAW event data.")
        return self._raw

    def event_window(self):
        """The real-world time of the start and end of event

        64-bit number since 1/1/1970
        """
        return (0.0, 0.0)

    def _get_peaks(self, peak_types, peak_class, sort):
        """Fetch S1 or S2 peaks from our data structure
        """
        # Sort key is used on the flattened peak
        # Sort order is whether or not Python sort is 'reversed'
        sort_key, sort_order = sort

        peaks = {}

        for peak in self._raw['peaks']:
            if peak['peak_type'] in peak_types:
                if 'rejected' in peaks and peaks['rejected'] is True:
                    continue

                # Flatten the peak so we can use our sort key
                peak_key = flatten(peak)[sort_key]

                # Save save into dictionary
                peaks[peak_key] = peak_class(peak)

        # Return just a list, but sorted according to our sort key and order
        return [peaks[i] for i in sorted(peaks, reverse=sort_order)]

    def S2s(self, sort=('top_and_bottom.area', True)):
        """List of S2 (ionization) signals as Peak objects"""
        return self._get_peaks(('large_s2', 'small_s2'), S2, sort)

    def S1s(self, sort=('top_and_bottom.area', True)):
        """List of S1 (scintillation) signals as Peak objects"""
        return self._get_peaks(('s1'), S1, sort)

    def pmt_waveform(self, pmt):
        """The individual waveform for a specific PMT"""
        if pmt not in self._raw['channel_waveforms'].keys():
            return None
        return self._raw['channel_waveforms'][pmt]

    def summed_waveform(self, name='top_and_bottom'):
        """Waveform summed over many PMTs"""
        if 'filtered' in name:
            raise ValueError('Use filtered_waveform, not this function: %s' % name)
        elif name not in self._raw['processed_waveforms']:
            self.log.debug(self._raw['processed_waveforms'].keys())
            raise ValueError("Summed waveform %s does not exist." % name)
        elif name == 'uncorrected_sum_waveform_for_xerawdp_matching':
            raise ValueError()

        return self._raw['processed_waveforms'][name]

    def filtered_waveform(self, filter_name=None):
        """Filtered waveform summed over many PMTs"""
        if filter_name is not None:
            #raise PendingDeprecationWarning() #Why? Currently this throws an error, not a warning...
            return self._raw['processed_waveforms'][filter_name]
        return self._raw['processed_waveforms']['filtered_for_large_s2']

    def dump(self):
        pprint.pprint(self._raw)


class Peak(object):
    def __init__(self, peak_dict):
        self.peak_dict = peak_dict

    def type(self):
        return self.__class__.__name__

    def _get_var(self, pmts, key):
        key = '%s.%s' % (pmts, key)
        flattened_peak = flatten(self.peak_dict)
        if key not in flattened_peak:
            pprint(self.peak_dict)
            raise ValueError('%s does not exist in peak' % key)

        return flattened_peak[key]

    def area(self, key='top_and_bottom'):
        return self._get_var(key, 'area')

    def width_fwhm(self, key='top_and_bottom'):
        return self._get_var(key, 'fwhm')

    def height(self, key='top_and_bottom'):
        return self._get_var(key, 'height')

    def time_in_waveform(self, key='top_and_bottom'):
        return self._get_var(key, 'position_of_max_in_waveform')

    def bounds(self):
        """Where the peak starts and ends in the sum waveform
        """
        return (self.peak_dict['left'],
                self.peak_dict['right'])


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
