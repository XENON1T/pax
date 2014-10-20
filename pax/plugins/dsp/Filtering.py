import numpy as np

from pax import plugin, datastructure


class Filtering(plugin.TransformPlugin):

    def startup(self):
        for f in self.config['filters']:
            # Check if the impulse response is sane
            ir = np.array(f['impulse_response'])
            if abs(1 - np.sum(ir)) > 0.0001:
                self.log.warning("Filter %s has non-normalized impulse response: %s != 1. Normalizing for you..." % (
                    f['name'], np.sum(ir))
                )
                ir /= np.sum(ir)
            if len(ir) % 2 == 0:
                self.log.warning("Filter %s has an even-length impulse response!" % f['name'])
            if not np.all(ir - ir[::-1] == 0):
                self.log.warning("Filter %s has an asymmetric impulse response!" % f['name'])

    def transform_event(self, event):
        for f in self.config['filters']:
            input_w = event.get_waveform(f['source'])
            output = np.convolve(input_w.samples, f['impulse_response'], 'same')

            if self.config['simulate_Xerawdp_convolution_bug']:
                ##
                # This dirty code hack implements the Xerawdp convolution bug
                # DO NOT USE except for Xerawdp matching!
                ##
                signal = input_w.samples
                filter_length = len(f['impulse_response'])
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

            event.waveforms.append(datastructure.Waveform({
                'name':      f['name'],
                'samples':   output,
                'pmt_list':  input_w.pmt_list,
            }))
        return event