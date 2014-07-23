#!/usr/bin/env python
from pax import pax

# Random notes:  expand on filter for small and large S2
# If plugins define docstring, we can load this in the docs
# Add issues about dependencies
# Explain what pruning is

if __name__ == '__main__':
    pax.processor(input='MongoDB.MongoDBInput',
                  transform=['DSP.JoinAndConvertWaveforms',
                             'DSP.ComputeSumWaveform',
                             'DSP.LargeS2Filter',
                             'DSP.SmallS2Filter',
                             #'DSP.PrepeakFinder',
                             #'DSP.FindPeaksInPrepeaks',
                             'DSP.FindS2_XeRawDPStyle',
                             'DSP.FindS1_XeRawDPStyle',
                             #'PeakPruning.PruneNonIsolatedPeaks',
                             'DSP.ComputeQuantities',
                             # 'PeakPruning.PruneWideShallowS2s',
                             # 'PeakPruning.PruneWideS1s',
                             # 'PeakPruning.PruneS1sWithNearbyNegativeExcursions',
                             'PeakPruning.PruneS1sInS2Tails',
                             'PeakPruning.PruneS2sInS2Tails'
                             #'PosSimple'
                             ],
                  output=['Plotting.PlottingWaveform',
                          #'Pickle.WriteToPickleFile'
                          ])
                          
