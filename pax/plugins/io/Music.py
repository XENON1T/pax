from scipy.io.wavfile import write
import numpy as np
from numpy import linspace,sin,pi,int16
from pylab import plot,show,axis
import scipy
import math

from pax.datastructure import Event
from pax import plugin

class WavOutput(plugin.OutputPlugin):
    """Convert sum waveforms of event and dataset to WAV file

    """

    def startup(self):
        # Used for building waveform for entire dataset
        self.all_data = {}

        self.start_time = None
        self.n = None
        self.rate = 44100

    def write_event(self, event):
        self.log.debug('Writing event')

        # Make sum waveform without PMT info
        data = event.pmt_waveforms.sum(0)
        self.n = len(data)

        # Note that // is an integer divide
        self.all_data[event.start_time//event.sample_duration] = data.copy()

        # Write a file
        write('song_%d.wav' % event.event_number,
              self.rate, # Frequency
              data)


    def shutdown(self):
        # Determine how long dataset music should be
        start = min(self.all_data.keys())
        end = max(self.all_data.keys())

        if self.n == None:
            self.log.error("Not known how many samples per event!")

        data = np.zeros(end - start + self.n, dtype=np.int16)
        for key, value in self.all_data.items():
            key -= start
            key = int(key)
            data[key : key + len(value)] += value


        # Resample such that data plays at live speed
        R = 10**8 // self.rate
        pad_size = math.ceil(float(data.size)/R)*R - data.size
        b_padded = np.append(data, np.zeros(pad_size)*np.NaN)
        new_data = scipy.nanmean(b_padded.reshape(-1,R), axis=1)

        # Output
        write('all.wav', self.rate, new_data)
        
        
