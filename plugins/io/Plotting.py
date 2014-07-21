import matplotlib.pyplot as plt
import random
import numpy as np

from pax import plugin


class PlottingWaveform(plugin.OutputPlugin):

    def write_event(self, event):
        """Plot an event

        Will make a fancy plot with lots of arrows etc of a summed waveform
        """
        self.log.debug("Making first plot")
        fig = plt.figure()
        ax = fig.add_subplot(211)

        side = 1
        # Plot all peaks
        for peak in event['peaks']:
            if peak['rejected']:
                continue
            x = peak['top_and_bottom']['position_of_max_in_waveform']
            y = event['processed_waveforms']['top_and_bottom'][x]

            plt.hlines(y, peak['left'], peak['right'])
            ax.annotate('%s:%s' % (peak['peak_type'], int(peak['top_and_bottom']['area'])),
                        xy=(x, y),
                        xytext=(peak['top_and_bottom']['position_of_max_in_waveform'],
                                event['processed_waveforms']['top_and_bottom'][
                                    peak['top_and_bottom']['position_of_max_in_waveform']] + 100 + random.random() * 100),
                        arrowprops=dict(arrowstyle="fancy",
                                        fc="0.6", ec="none",
                                        connectionstyle="angle3,angleA=0,angleB=-90"))
            side *= -1

        plt.plot(event['processed_waveforms']['uncorrected_sum_waveform_for_xerawdp_matching'], label='uncorrected')
        plt.plot(event['processed_waveforms']['top_and_bottom'], label='top_and_bottom')
        plt.plot(event['processed_waveforms']['filtered_for_large_s2'],
                 '--', label='filtered_for_large_s2')
        plt.plot(event['processed_waveforms']['filtered_for_small_s2'],
                 '--', label='filtered_for_small_s2')

        plt.legend()
        plt.xlabel('Time in event [10 ns]')
        plt.ylabel("pe / bin")


class PlottingHitPattern(plugin.OutputPlugin):

    def startup(self):
        self.topArrayMap = self.config['topArrayMap']

    def write_event(self, event):
        """Plot an event

        Will make a fancy plot with lots of arrows etc of a summed waveform
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        x = []
        y = []
        area = []

        # need function to get PMT location...
        for pmt, data in event['channel_waveforms'].items():
            if pmt not in self.topArrayMap.keys():
                continue
            pmt_location = self.topArrayMap[pmt]
            indices = np.flatnonzero(data)

            x.append(pmt_location['x'])
            y.append(pmt_location['y'])
            area.append(np.sum(data[indices]))

        area = np.array(area) / 10

        ax = plt.subplot(111)
        c = plt.scatter(x, y, c='red',
                        s=area, cmap=plt.cm.hsv)
        c.set_alpha(0.75)

        plt.show(block=False)

        self.log.info("Hit enter to continue...")
        input()
