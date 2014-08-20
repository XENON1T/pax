import numpy as np

from pax import plugin

from pax.datastructure import Waveform


class GenericFilter(plugin.TransformPlugin):

    """Generic filter base class

    Do not instantiate. Instead, subclass: subclass has to set
        self.filter_ir  --  filter impulse response (normalized, i.e. sum to 1)
        self.input_name      --  label of waveform in processed_waveforms to filter
        self.output_name     --  label where filtered waveform is stored in processed_waveforms

    TODO: Add some check the name of the class and throw exception if base class
          is instantiated.  use self.name.
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
        if round(sum(self.filter_ir), 5) != 1.:
            raise RuntimeError('Impulse response sums to %s, should be 1!' % sum(self.filter_ir))
        input_w = event.get_waveform(self.input_name)
        signal = input_w.samples
        filter_length = len(self.filter_ir)
        # Apply the filter
        output = np.array(np.convolve(signal, self.filter_ir, 'same'), dtype=np.float64)
        if 'simulate_Xerawdp_convolution_bug' in self.config and self.config['simulate_Xerawdp_convolution_bug']:
            ##
            # Mutilate waveform for Xerawdp matching
            # This implements the Xerawdp convolution bug
            ##
            # Determine the pulse boundaries
            y = np.abs(np.sign(signal))
            pbs = np.concatenate((np.where(np.roll(y, 1) - y == -1)[0], np.where(np.roll(y, -1) - y == -1)[0]))
            # Check if these are real pulse boundaries: at least three samples before or after must be zero
            real_pbs = []
            for q in pbs:
                if q < 3 or q > len(signal) - 4:
                    continue  # So these tests don't fail
                if signal[q - 1] == signal[q - 2] == signal[q - 3] == 0 or signal[q + 1] == signal[q + 2] == signal[q + 3] == 0:
                    real_pbs.append(q)
            # event['pulse_boundaries'][self.input_name] = real_pbs
            # Mutilate the waveform
            # First mutilate the edges, which are always pulse boundaries
            output[:int(filter_length / 2)] = np.zeros(int(filter_length / 2))
            output[len(output) - int(filter_length / 2):] = np.zeros(int(filter_length / 2))
            # Mutilate waveform around pulse boundaries
            for pb in real_pbs:
                try:
                    lefti = max(0, pb - int(filter_length / 2))
                    righti = min(len(signal) - 1, pb + int(filter_length / 2))
                    output[lefti:righti] = np.zeros(righti - lefti)
                except Exception as e:
                    self.log.warning("Error during waveform mutilation: " + str(e) + ". So what...")
        # Store the result
        event.waveforms.append(Waveform({
            'samples':  output,
            'name':     self.output_name,
            'pmt_list': input_w.pmt_list
        }))
        return event


class LargeS2Filter(GenericFilter):

    """Docstring  Low-pass filter using raised cosine filter

    TODO: put constants into ini?
    """

    def startup(self):
        GenericFilter.startup(self)

        #self.filter_ir = self.rcosfilter(31, 0.2, 3 * units.MHz * self.config['digitizer_t_resolution'])
        # Guillaume's raised cosine coeffs:
        self.filter_ir = [0.005452,  0.009142,  0.013074,  0.017179,  0.021381,  0.025597,  0.029746,  0.033740,  0.037499,  0.040941,  0.043992,  0.046586,  0.048666,  0.050185,  0.051111,
                          0.051422,  0.051111,  0.050185,  0.048666,  0.046586,  0.043992,  0.040941,  0.037499,  0.033740,  0.029746,  0.025597,  0.021381,  0.017179,  0.013074,  0.009142,  0.005452]
        self.output_name = 'filtered_for_large_s2'
        self.input_name = 'uS2'

    @staticmethod
    def rcosfilter(filter_length, rolloff, cutoff_freq, sampling_freq=1):
        """
        Returns a nd(float)-array describing a raised cosine (RC) filter (FIR) impulse response. Arguments:
            - filter_length:    filter event_duration in samples
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
        self.input_name = 'uS2'


class S1WidthTestFilter(GenericFilter):

    """

    TODO: take this opportunity to explain why there is a small s2 filter... even if it stupid.
    TODO: put constants into ini?
    """

    def startup(self):
        GenericFilter.startup(self)
        # Yeah, these are the same as the large s2 filter, I know
        # But the input waveform is different, it has some excluded PMTs
        self.filter_ir = np.array([0.005452,  0.009142,  0.013074,  0.017179,  0.021381,  0.025597,  0.029746,  0.033740,  0.037499,  0.040941,  0.043992,  0.046586,  0.048666,  0.050185,
                                   0.051111,  0.051422,  0.051111,  0.050185,  0.048666,  0.046586,  0.043992,  0.040941,  0.037499,  0.033740,  0.029746,  0.025597,  0.021381,  0.017179,  0.013074,  0.009142,  0.005452])
        self.filter_ir = self.filter_ir / sum(self.filter_ir)  # Normalization
        self.output_name = 'filtered_for_s1_width_test'
        self.input_name = 'uS1'
