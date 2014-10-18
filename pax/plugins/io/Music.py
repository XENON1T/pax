from scipy.io.wavfile import write
import numpy as np
from numpy import linspace,sin,pi,int16
from pylab import plot,show,axis
import scipy
import math

from pax.datastructure import Event
from pax import plugin

class WavOutput(plugin.OutputPlugin):
    def note(freq, len, amp=1, rate=44100):
        # tone synthesis  
        t = linspace(0,len,len*rate)
        data = sin(2*pi*freq*t)*amp
        return data.astype(int16) # two byte integers

    def startup(self):
        self.all_data = {}

        self.start_time = None
        

    def write_event(self, event):
        self.log.debug('Writing event')


        data = event.pmt_waveforms.sum(0)

        self.all_data[event.start_time//event.sample_duration] = data.copy()

        # A tone, 2 seconds, 44100 samples per second                                                                                                                                       
        #tone = note(440,2,amp=10000)

        #write('440hzAtone.wav',44100,tone) # writing the sound to a file                                                                                                                    

        #plot(linspace(0,2,2*44100),tone)
        #axis([0,0.4,15000,-15000])
        #show()

        write('song_%d.wav' % event.event_number,
              44100,
              data)

    def shutdown(self):
        start = min(self.all_data.keys())
        end = max(self.all_data.keys())

        print(start, end)

        data = np.zeros(end - start + 40000, dtype=np.int16)
        for key, value in self.all_data.items():
            key -= start
            key = int(key)
            data[key : key + len(value)] += value
            print(key, value)

        write('all.wav', 10**8, data)
        
        R = 10**8 // 44100
        pad_size = math.ceil(float(data.size)/R)*R - data.size
        b_padded = np.append(data, np.zeros(pad_size)*np.NaN)
        new_data = scipy.nanmean(b_padded.reshape(-1,R), axis=1)
        
        write('all2.wav', 44100, new_data)
        
        
