#!/usr/bin/env python
from pax import pax

if __name__ == '__main__':
    pax.Processor(input='MongoDB.MongoDBInput',
                  transform=['DSP.JoinAndConvertWaveforms',
                             'DSP.ComputeSumWaveform',
                             'DSP.LargeS2Filter',
                             'DSP.SmallS2Filter',
                             'DSP.LargeS2Peakfinder',
                             'DSP.SmallS2Peakfinder',
                             'DSP.S1Peakfinder',
                             #'DSP.VetoS1Peakfinder',
                             'DSP.ComputeQuantities'],
                  output=['PlottingWaveform.PlottingWaveform',
                          'Pickle.WriteToPickleFile'])

