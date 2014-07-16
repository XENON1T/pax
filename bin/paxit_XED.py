#!/usr/bin/env python
from pax import pax

if __name__ == '__main__':
    pax.processor(input='XED.ReadXed',
                  transform=['DSP.JoinAndConvertWaveforms',
                             'DSP.ComputeSumWaveform',
                             'DSP.LargeS2Filter',
                             'DSP.SmallS2Filter',
                             'DSP.LargeS2Peakfinder',
                             'DSP.SmallS2Peakfinder',
                             'DSP.S1Peakfinder',
                             # 'DSP.VetoS1Peakfinder',
                             'DSP.ComputeQuantities',
                             'PeakPruning.PruneWideShallowS2s',
                             'PeakPruning.PruneWideS1s',
                             'PeakPruning.PruneS1sWithNearbyNegativeExcursions',
                             'PeakPruning.PruneS1sInS2Tails',
                             'PeakPruning.PruneS2sInS2Tails'
                             ],
                  output=['PlottingWaveform.PlottingWaveform',
                          #'Pickle.WriteToPickleFile',
                          #'CSV.WriteCSVPeakwise'
                          ])
