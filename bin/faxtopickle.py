#!/usr/bin/env python
from pax import pax

config_overload = """
[pax]
input = 'fax.FaX'
output = 'Pickle.WriteToPickleFile'
dsp = []
transform = []

[Pickle.WriteToPickleFile]
output_dir = './pickled'

[fax.FaX]
instruction_file_filename =           'peakfinder_testbank.csv'
magically_avoid_dead_pmts =           True
magically_avoid_s1_excluded_pmts =    True
event_repetitions =                   100   # Simulate each event in the instruction file this many times (1 means: simulate just once, no repetitions)

s1_detection_efficiency   =           1

pad_before =                          20*us               # Padding before a peak
pad_after =                           20*us               #         after
event_padding =                       0                   # Zero-padding on either side of event (no noise)

white_noise_sigma =                   0.5 * uA

"""

if __name__ == '__main__':
    pax.processor(config_overload)
