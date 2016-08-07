import numpy as np
import numba

from pax.trigger import TriggerPlugin, pulse_dtype


class SortData(TriggerPlugin):
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

        # Find the sort order of the data.
        sort_order = np.argsort(ind['start_times'])

        # Build the structured array with the data
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

        # Perform the sort with the order determined previously.
        # See http://stackoverflow.com/questions/19682521/sorting-numpy-structured-and-reco
        # Especially the comment on the answer on why the mode should always be 'raise'!
        np.take(pulses, sort_order, out=pulses, mode='raise')
        data.pulses = pulses
        del data.input_data


@numba.jit(nopython=True)
def get_pmt_numbers(channels, modules, pmts_buffer, pmt_lookup):
    """Fills pmts_buffer with pmt numbers corresponding to channels, modules according to pmt_lookup matrix:
     - pmt_lookup: lookup matrix for pmt numbers. First index is digitizer module, second is digitizer channel.
    Modifies pmts_buffer in-place.
    """
    for i in range(len(channels)):
        pmts_buffer[i] = pmt_lookup[modules[i], channels[i]]
