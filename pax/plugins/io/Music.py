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

        self.n = None
        self.rate = 40000
        self.single_events = True

    def write_event(self, event):
        self.log.debug('Writing event')

        # Make sum waveform without PMT info
        data = event.pmt_waveforms.sum(0)
        self.n = len(data)

        # Note that // is an integer divide
        self.all_data[event.start_time] = event.pmt_waveforms.sum()

        if self.single_events:
            # Write a file
            s1 = event.get_waveform('s1_peakfinding').samples
            s2 = event.get_waveform('filtered_for_large_s2').samples

            duration = 0.01 # seconds
            n = s1.size

            f1 = 1046.50 # Hz
            f2 = 130.813 # Hz

            s1 = np.repeat(s1, duration * self.rate)
            s2 = np.repeat(s2, duration * self.rate)

            t = np.linspace(0,
                            duration * n,
                            duration * self.rate * n)

            d1 = np.sin(2 * np.pi * f1 * t) * s1
            d2 = np.sin(2 * np.pi * f2 * t) * s2
            data = (d1 + d2)

            print(data.sum())
            write('song_%d.wav' % event.event_number,
                  self.rate, # Frequency
                  data)


    def shutdown(self):
        if self.single_events:
            return
        
        start = min(self.all_data)
        end = max(self.all_data)

        # Determine how long dataset music should be
        self.log.info(end - start)

        if self.n == None:
            self.log.error("Not known how many samples per event!")

        # Resample such that data plays at live speed
        R = 10**9 // self.rate

        n = math.ceil((end - start) / R)
        data = np.zeros(n, dtype=np.int16)
        for key, value in self.all_data.items():
            key -= start
            key = int(key // R)

            if value > 0:
                data[key] = np.min([value, 2**15 -1])

        # Output
        write(self.filename,
              self.rate,
              data)
        
        
