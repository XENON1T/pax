""""
This class will plot all waveforms that have been recorded. It reads the created pulses for each event
and plot the waveforms in a single plot
"""

import matplotlib as plt

import pax.plugins.plotting.Plotting


class ShowWaveforms(pax.PlotBase):
    def PlotAllChannels(self, event):
        fig, ax = plt.subplots(nrows=7, ncols=2, sharex=True, sharey=True, squeeze=False, figsize=(12, 12))
        plt.subplots_adjust(hspace=0, wspace=0.05)
        for xi in range(7):
            for yi in range(2):
                ax[xi][yi].plot(range(10), range(10))
