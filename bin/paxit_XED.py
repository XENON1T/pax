#!/usr/bin/env python
from pax import pax

config_overload = """
[pax]
input = 'XED.XedInput'
#output = 'Plotting.PlottingWaveform'
output = 'CSV.WriteCSVPeakwise'

[XED.XedInput]
filename = "xe100_120402_2000_000000.xed"
"""

if __name__ == '__main__':
    pax.processor(config_overload)