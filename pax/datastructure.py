import logging
import inspect
__author__ = 'tunnell'

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

    @property
    def event_window(self):
        """The real-world time of the start and end of event

        64-bit number since 1/1/1970
        """
        return (0.0, 0.0)

    @property
    def get_position(self, recon_algo='nn'):
        """3-vector position in mm
        """
        x, y, _ = self.S2s(sort='area')[0].position
        _, _, z = self.S1s(sort='coincidence')[0].position
        return (0.0, 0.0, 0.0)

    @property
    def S2s(self, sort=(1, 'area')):
        """List of S2 (ionization) signals as Peak objects"""
        peaks = []
        # Check sorting
        peak = S2({})
        for peak in self.raw['peaks']:
            if peak['input'] == 'filtered_for_large_s2' or peak['input'] == 'filtered_for_small_s2':
                print(peak)
        peaks.append(peak)
        return peaks

    @property
    def S1s(self, sort=(1, 'area')):
        """List of S1 (scintillation) signals as Peak objects"""
        return None

    def pmt_waveform(self, pmt):
        """The individual waveform for a specific PMT"""
        return self.raw.channel_waveforms[pmt]

    @property
    def summed_waveform(self, name='top_and_bottom'):
        """Waveform summed over many PMTs"""
        if 'filtered' in name:
            raise ValueError('Do not get filtered waveforms with summed waveform; use filtered_waveform: %s' % name)
        elif name not in self.raw['processed_waveform']:
            raise ValueError("Summed waveform %s does not exist")

        return self.raw['processed_waveform'][name]

    @property
    def filtered_waveform(self):
        """Filtered waveform summed over many PMTs"""
        return self.raw['processed_waveform']['filtered_for_large_s2']

    @property
    def event_attributes(self):
        """Python only extra event attributes"""
        PendingDeprecationWarning()
        return None

    def explain(self):
        x = inspect.getmembers(self.__class__,
                               predicate=inspect.isdatadescriptor)

        for a, b in x:
            if a.startswith('_'):
                continue
            print(a, b.__doc__)

    def dump(self):
        import json
        import pprint
        import numpy as np

        class NumpyAwareJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray) and obj.ndim == 1:
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        pprint.pprint(self.raw)
        json.dumps(self.raw, cls=NumpyAwareJSONEncoder)


class Peak(object):
    def __init__(self, peak_dict):
        self.peak_dict = {}

    def position(self, recon='name'):
        if recon in self.recon.keys():
            return self.recon[recon]

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


