"""
Utility functions for digital signal processing
"""
import numpy as np


def baseline_mean_stdev(waveform, sample_size=46):
    """ returns (baseline, baseline_stdev), calculated on the first sample_size samples of waveform """
    baseline_sample = waveform[:sample_size]
    return (
        np.mean(sorted(baseline_sample)[
            int(0.4*len(baseline_sample)):int(0.6*len(baseline_sample))
        ]),  #Ensures peaks in baseline sample don't skew computed baseline
        np.std(baseline_sample)                     #... but do count towards baseline_stdev!
    )
    #Don't want to just take the median as V-resolution is finite
    #Don't want the mean either: this is not robust against large fluctuations (eg peaks in sample)

def extent_until_treshold(signal, start, treshold):
    a = interval_until_treshold(signal, start, treshold)
    return a[1]-a[0]

def interval_until_treshold(signal, start, treshold):
    return (
        find_first_below(signal, start, treshold, 'left'),
        find_first_below(signal, start, treshold, 'right'),
    )
    
def find_first_below(signal, start, treshold, direction, min_length_below=1):
    #TODO: test for off-by-one errors
    counter = 0
    if direction=='right':
        for i,x in enumerate(signal[start:]):
            if x<treshold: return start+i
    elif direction=='left':
        i = start
        while 1:
            if signal[i]<treshold:
                counter += 1
                if counter == min_length_below:
                    return i    #or i-min_length_below ??? #TODO
            else:
                counter = 0
            if direction=='right':
                i += 1
            elif direction=='left':
                i -= 1
            else:
                raise(Exception, "You nuts? %s isn't a direction!" % direction)
                
def all_same_length(items): return all(len(x) == len(items[0]) for x in items) 