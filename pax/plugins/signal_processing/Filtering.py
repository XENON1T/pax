import numpy as np

from pax import plugin, datastructure, units
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import freqz


class Filtering(plugin.TransformPlugin):

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def startup(self):

        if self.config['simulate_Xerawdp_convolution_bug']:
            self.log.warning('Xerawdp convolution bug simulation is enabled: S2 peak finding will be less accurate.')

        for f in self.config['filters']:
            self.log.debug("Applying filter %s" % f['name'])

            if 'impulse_response' in f:
                # Time domain filter: convolution with an impulse response
                ir = np.array(f['impulse_response'])

                # Check if the impulse response is sane

                if abs(1 - np.sum(ir)) > 0.0001:
                    self.log.warning("Filter %s has non-normalized impulse response: %s != 1. Normalizing for you..." % (
                        f['name'], np.sum(ir))
                    )
                    f['impulse_response'] = ir/np.sum(ir)

                if len(ir) % 2 == 0:
                    self.log.warning("Filter %s has an even-length impulse response!" % f['name'])

                if not np.all(ir - ir[::-1] == 0):
                    self.log.warning("Filter %s has an asymmetric impulse response!" % f['name'])

            else:
                # Frequency bandpass filter
                # Implementation from http://wiki.scipy.org/Cookbook/ButterworthBandpass
                sampling_rate = 1/self.config['digitizer_t_resolution']

                # # Plot the frequency response for a few different orders.
                # fs = sampling_rate
                # import matplotlib.pyplot as plt
                # plt.figure(1)
                # plt.clf()
                # for order in range(6):
                #     b, a = self.butter_bandpass(
                #          lowcut=f['low_freq_bound'],
                #          highcut=f['high_freq_bound'],
                #          fs=sampling_rate,
                #          order=order
                #     )
                #     w, h = freqz(b, a, worN=2000)
                #     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
                #
                # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
                #          '--', label='sqrt(0.5)')
                # plt.xlabel('Frequency (GHz)')
                # plt.ylabel('Gain')
                # plt.grid(True)
                # plt.legend(loc='best')

                f['filter_parameters'] = self.butter_bandpass(
                    lowcut=f['low_freq_bound'],
                    highcut=f['high_freq_bound'],
                    fs=sampling_rate,
                    order=1    # At high orders the frequency response seems to go bananas..?
                )


    def transform_event(self, event):
        for f in self.config['filters']:

            input_w = event.get_waveform(f['source'])
            signal = input_w.samples
            if 'impulse_response' in f:
                output = np.convolve(signal, f['impulse_response'], 'same')
            else:
                output = filtfilt(f['filter_parameters'][0], f['filter_parameters'][1], signal)

            if self.config['simulate_Xerawdp_convolution_bug'] and 'impulse_response' in f:
                ##
                # This dirty code hack implements the Xerawdp convolution bug
                # DO NOT USE except for Xerawdp matching!
                ##
                #TODO: could be done more straightforwardly now that we've stored occurrences properly
                filter_length = len(f['impulse_response'])

                # Determine the pulse boundaries
                y = np.abs(np.sign(signal))
                pbs = np.concatenate((np.where(np.roll(y, 1) - y == -1)[0],
                                      np.where(np.roll(y, -1) - y == -1)[0]))

                # Check if these are real pulse boundaries: at least three samples before or after must be zero
                real_pbs = []
                for q in pbs:
                    if q < 3 or q > len(signal) - 4:
                        continue  # So these tests don't fail
                    if signal[q - 1] == signal[q - 2] == signal[q - 3] == 0 or signal[q + 1] == signal[q + 2] == signal[q + 3] == 0:
                        real_pbs.append(q)

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

            event.waveforms.append(datastructure.SumWaveform({
                'name':      f['name'],
                'samples':   output,
                'pmt_list':  input_w.pmt_list,
                'detector':  input_w.detector,
            }))
        return event
