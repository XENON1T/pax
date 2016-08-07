import numpy as np
import numba

from pax.recarray_tools import dict_group_by
from pax.trigger import TriggerPlugin, pulse_dtype


class MakePulsesArray(TriggerPlugin):
    """Converts the raw data given to trigger.run into a nice numpy array,
    looking up PMT numbers for module/channel pairs, and sorting the array by time.
    """

    def startup(self):
        # Build a (module, channel) ->  lookup matrix
        # I wish numba had some kind of dictionary / hashtable support...
        # but this will work as long as the module serial numbers are small :-)
        # I will assume always and everywhere the pmt position numbers start at 0 and increase by 1 continuously!
        # Initialize the matrix to n_channels, which is one above the last PMT
        # This will ensure we do not crash on data in 'ghost' channels (not plugged in,
        # do report data in self-triggered mode)
        pmt_data = self.trigger.pax_config['DEFAULT']['pmts']
        self.n_channels = len(pmt_data)
        max_module = max([q['digitizer']['module'] for q in pmt_data])
        max_channel = max([q['digitizer']['channel'] for q in pmt_data])
        self.pmt_lookup = self.n_channels * np.ones((max_module + 1, max_channel + 1), dtype=np.int)
        for q in pmt_data:
            module = q['digitizer']['module']
            channel = q['digitizer']['channel']
            self.pmt_lookup[module][channel] = q['pmt_position']

    def process(self, data):
        ind = data.input_data
        # Enter the times into data
        pulses = np.zeros(len(ind['start_times']), dtype=pulse_dtype)
        pulses['time'] = ind['start_times']
        if ind['channels'] is not None and ind['modules'] is not None:
            # Convert the channel/module specs into pmt numbers.
            get_pmt_numbers(ind['channels'], ind['modules'], pmts_buffer=pulses['pmt'], pmt_lookup=self.pmt_lookup)
        else:
            # If PMT numbers are not specified, pretend everything is from a 'ghost' pmt at # = n_channels
            pulses['pmt'] = self.n_channels
        if ind['areas'] is not None:
            pulses['area'] = ind['areas']
        data.pulses = pulses
        data.pulses.sort(order='time')
        del data.input_data


class SortByDetectorAndTime(TriggerPlugin):
    """Sort pulses by detector (in data.pulses_per_detector), and for each detector by time"""

    def startup(self):
        # Build a channel -> detector lookup list
        self.detectors = []    # List of detectors in some order (arbitrary since channels_in_detector is a dict field)
        detector_lookup = {}   # dict mapping ch -> detector index in self.detectors
        for i, (detector, chs) in enumerate(self.trigger.pax_config['DEFAULT']['channels_in_detector'].items()):
            self.detectors.append(detector)
            for ch in chs:
                detector_lookup[ch] = i
        self.detector_lookup = np.array([detector_lookup[ch] for ch in range(len(detector_lookup))])

    def process(self, data):
        pulses = data.pulses
        get_detector_numbers(pulses['pmt'], pulses['_detector'], self.detector_lookup)
        data.pulses.sort(order=['_detector', 'time'])
        for detector_number, ts in dict_group_by(pulses, '_detector').items():
            data.pulses_per_detector[self.detectors[detector_number]] = ts
        del data.pulses


@numba.jit(nopython=True)
def get_pmt_numbers(channels, modules, pmts_buffer, pmt_lookup):
    """Fills pmts_buffer with pmt numbers corresponding to channels, modules according to pmt_lookup matrix:
     - pmt_lookup: lookup matrix for pmt numbers. First index is digitizer module, second is digitizer channel.
    Modifies pmts_buffer in-place.
    """
    for i in range(len(channels)):
        pmts_buffer[i] = pmt_lookup[modules[i], channels[i]]


@numba.jit(nopython=True)
def get_detector_numbers(pmt_numbers, detectors_buffer, detector_lookup):
    """Fills detectors_buffer with detector numbers corresponding to pmt_numbers according to detector_lookup array.
    Modifies detectors_buffer in-place.
    """
    for i in range(len(pmt_numbers)):
        detectors_buffer[i] = detector_lookup[pmt_numbers[i]]
