#!/usr/bin/env python
from pax import pax

# Random notes:  expand on filter for small and large S2
# If plugins define docstring, we can load this in the docs
# Add issues about dependencies
# Explain what pruning is

config_overload = """
[MongoDB.MongoDBInput]
collection = "dataset"
address = "127.0.0.1:27017"
"""

if __name__ == '__main__':
    pax.processor(input='MongoDB.MongoDBInput',
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
                             'PeakPruning.PruneS2sInS2Tails',
                             'DatastructureConverter.ConvertToEventClass'
                             ],
                  output=['Plotting.PlottingWaveform', # Make plot using matplotlib
                          'Pickle.WriteToPickleFile',  # Write to Pythonistic file
                          #'Display.DisplayServer',     # Start web server for plots
                          ],
                  config_overload=config_overload,
                  )

