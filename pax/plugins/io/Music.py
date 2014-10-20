from scipy.io.wavfile import write
import numpy as np

import math
from pax import plugin


class WavOutput(plugin.OutputPlugin):
    """Convert sum waveforms of event and dataset to WAV file

    If we don't find dark matter, at least we'll have contemporary music.
    """

    def startup(self):
        self.filename = self.config['wav_file']

        # Used for building waveform for entire dataset
        self.all_data = {}

        self.start_time = None
        self.n = None
        self.rate = 40000
        self.single_events = False

    def write_event(self, event):
        self.log.debug('Writing event')

        # Make sum waveform without PMT info
        data = event.pmt_waveforms.sum(0)
        self.n = len(data)

        # Note that // is an integer divide
        self.all_data[event.start_time] = event.pmt_waveforms.sum()

        if self.single_events:
            # Write a file
            write('song_%d.wav' % event.event_number,
                  self.rate, # Frequency
                  data)


    def shutdown(self):
        # Determine how long dataset music should be
        start = min(self.all_data)
        end = max(self.all_data)

        if self.n == None:
            self.log.error("Not known how many samples per event!")

        # Resample such that data plays at live speed
        R = 10**9 // self.rate

        n = math.ceil((end - start) / R)
        data = np.zeros(n, dtype=np.int16)
        for key, value in self.all_data.items():
            key = int(key // R)
            if value > 0:
                data[key] = math.min(value, 2**14)

        # Output
        write(self.filename,
              self.rate,
              data)
        
        
