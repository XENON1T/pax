#!/usr/bin/env python
from pax import pax

if __name__ == '__main__':
    pax.processor(input='XED.XedInput',
                  transform=['DSP.JoinAndConvertWaveforms',
                             'DSP.ComputeSumWaveform',
                             'DSP.LargeS2Filter',
                             'DSP.SmallS2Filter',
                             'DSP.PrepeakFinder',
                             'DSP.FindPeaksInPrepeaks',
                             'DSP.FindS1_XeRawDPStyle',
                             'PeakPruning.PruneNonIsolatedPeaks',
                             'DSP.ComputeQuantities',
                             'PeakPruning.PruneWideShallowS2s',
                             # 'PeakPruning.PruneWideS1s',
                             # 'PeakPruning.PruneS1sWithNearbyNegativeExcursions',
                             'PeakPruning.PruneS1sInS2Tails',
                             'PeakPruning.PruneS2sInS2Tails'
                             ],
                  output=['Plotting.PlottingWaveform',
                          #'Pickle.WriteToPickleFile',
                          #'CSV.WriteCSVPeakwise'
                          ])
