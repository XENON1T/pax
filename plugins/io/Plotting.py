import matplotlib.pyplot as plt
import random
import numpy as np

from pax import plugin


class PlottingWaveform(plugin.OutputPlugin):

    def write_event(self, event):
        """Plot an event

        Will make a fancy plot with lots of arrows etc of a summed waveform
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)


        # Plot all peaks
        for peak in event.S2s() + event.S1s():

            x = peak.time_in_waveform()
            y = event.summed_waveform()[x]

            plt.hlines(y, *peak.bounds())
            ax.annotate('%s:%s' % (peak.type, int(peak.area())),
                        xy=(x, y),
                        xytext=(x, y + 100 + random.random() * 100),
                        arrowprops=dict(arrowstyle="fancy",
                                        fc="0.6", ec="none",
                                        connectionstyle="angle3,angleA=0,angleB=-90"))
            side *= -1

        plt.plot(event.summed_waveform('uncorrected_sum_waveform_for_xerawdp_matching'),
                 label='uncorrected') # Deprecation warning
        plt.plot(event.summed_waveform(), label='summed')
        plt.plot(event.filtered_waveform(), '--', label='filtered')
        plt.plot(event.filtered_waveform('filtered_for_small_s2'), '--',
                 'filtered_for_small_s2') # Pending deprecation


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

        for pmt, pmt_location in self.topArrayMap.keys():
            q = np.sum(event.pmt_waveform(pmt))

            x.append(pmt_location['x'])
            y.append(pmt_location['y'])
            area.append(q)

        area = np.array(area) / 10

        ax = plt.subplot(111)
        c = plt.scatter(x, y, c='red',
                        s=area, cmap=plt.cm.hsv)
        c.set_alpha(0.75)

        plt.show(block=False)

        self.log.info("Hit enter to continue...")
        input()
