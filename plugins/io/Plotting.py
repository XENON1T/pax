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
        max_y = max([p.height for p in event.S2s + event.S1s])

        for peak in event.S2s + event.S1s:

            x = peak.index_of_maximum
            y = peak.height

            plt.hlines(y, peak.left, peak.right)
            ax.annotate('%s:%s' % (('s1' if hasattr(peak, 'is_s1') else 's2'), int(peak.area)),
                        xy=(x, y),
                        xytext=(x, y + (max_y-y)*(0.05+0.2*random.random())),
                        arrowprops=dict(arrowstyle="fancy",
                                        fc="0.6", ec="none",
                                        connectionstyle="angle3,angleA=0,angleB=-90"))

        # plt.plot(event.get_waveform('tpc').samples,  label='TPC')
        # plt.plot(event.get_waveform('filtered_for_s2').samples, label='TPC - filtered')
        # plt.plot(event.get_waveform('veto').samples, label='Veto')
        plt.plot(event.get_waveform('uS1').samples, label='S1 peakfinding')
        plt.plot(event.get_waveform('uS2').samples, label='S2 peakfinding')
        plt.plot(event.get_waveform('filtered_for_large_s2').samples, label='Large s2 filtered')
        plt.plot(event.get_waveform('filtered_for_small_s2').samples, label='Small s2 filtered')
        plt.plot([0.6241506363] * event.length(),  '--', label='Large S2 threshold')
        plt.plot([0.06241506363]* event.length(),  '--', label='Small S2 threshold')
        plt.plot([0.1872451909] * event.length(),  '--', label='S1 threshold')

        legend = plt.legend(loc='upper left', prop={'size':10})
        legend.get_frame().set_alpha(0.5)
        plt.xlabel('Sample in event [10 ns]')
        plt.ylabel("pe / bin")
        plt.tight_layout()

        plt.show(block=False)
        self.log.info("Hit enter to continue...")
        input()
        plt.close()


class PlottingHitPattern(plugin.OutputPlugin):
    def startup(self):
        self.topArrayMap = self.config['topArrayMap']

    def write_event(self, event):
        """Plot an event

        Will make a fancy plot with lots of arrows etc of a summed waveform
        """
        plt.figure()

        x = []
        y = []
        area = []

        for pmt, pmt_location in self.topArrayMap.keys():
            q = np.sum(event.pmt_waveform(pmt))

            x.append(pmt_location['x'])
            y.append(pmt_location['y'])
            area.append(q)

        area = np.array(area) / 10

        c = plt.scatter(x, y, c='red',
                        s=area, cmap=plt.cm.hsv)
        c.set_alpha(0.75)

        plt.show(block=False)

        self.log.info("Hit enter to continue...")
        input()
