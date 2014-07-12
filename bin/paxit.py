#!/usr/bin/env python
from pax import pax

if __name__ == '__main__':
    pax.Processor(input='MongoDB.MongoDBInput',
                  transform=['DSP.ComputeSumWaveform',
                             'DSP.FilterWaveforms'],
                  output='PlottingWaveform.PlottingWaveform')

