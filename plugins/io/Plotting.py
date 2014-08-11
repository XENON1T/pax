import matplotlib.pyplot as plt
import random
import numpy as np
import os

from pax import plugin, units


class PlottingWaveform(plugin.OutputPlugin):

    def startup(self):
        self.plt = plt
        if 'output_dir' in self.config:
            self.output_dir = self.config['output_dir']
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        else:
            self.output_dir = None

    def write_event(self, event):
        """Plot an event

        Will make a fancy plot with lots of arrows etc of a summed waveform
        """
        self.time_conversion_factor = event.sample_duration * units.ns / units.us
        plt = self.plt
        rows = 2
        cols = 3
        plt.figure(figsize=(6 * cols, 4 * rows))
        # Plot the largest S1 and largest S2 in the event
        if event.S1s:
            ax = plt.subplot2grid((rows, cols), (0, 0))
            largest_s1 = sorted(event.S1s, key=lambda x: x.area, reverse=True)[0]
            self.plot_waveform(event, left=largest_s1.left, right=largest_s1.right, pad=10)
            plt.title("S1 at %.1f us" % (largest_s1.index_of_maximum * self.time_conversion_factor))
        if event.S2s:
            ax = plt.subplot2grid((rows, cols), (0, 1))
            ax.yaxis.set_label_position("right")
            largest_s2 = sorted(event.S2s, key=lambda x: x.area, reverse=True)[0]
            self.plot_waveform(event, left=largest_s2.left, right=largest_s2.right, pad=50)
            plt.title("S2 at %.1f us" % (largest_s2.index_of_maximum * self.time_conversion_factor))

        #plt.subplot2grid((rows,cols), (0,2))
        #plt.title('Event %s from %s' % (event.event_number, 'mysterious_dataset'))
        # Todo: plot hitmap of s2

        # Plot the total waveform
        plt.subplot2grid((rows, cols), (rows - 1, 0), colspan=cols)
        self.plot_waveform(event, show_peaks=True)
        plt.yscale('log', nonposy='clip')
        plt.ylim(10 ** (-2), plt.ylim()[1])
        legend = plt.legend(loc='upper left', prop={'size': 10})
        legend.get_frame().set_alpha(0.5)
        plt.tight_layout()
        if self.output_dir:
            plt.savefig(self.output_dir + '/' + str(event.event_number) + '.png')
        else:
            plt.show(block=False)
            self.log.info("Hit enter to continue...")
            input()
            plt.close()

    def plot_waveform(self, event, left=0, right=None, pad=0, show_peaks=False):
        if right is None:
            right = event.length() - 1
        lefti = left - pad
        righti = right + pad
        nsamples = 1 + righti - lefti
        dt = event.sample_duration * units.ns
        xlabels = np.arange(
            lefti * dt / units.us,
            (10 + righti) * dt / units.us,  # 10+ labels will be cut later, prevents off by one errors
            dt / units.us
        )
        xlabels = xlabels[:nsamples]
        plt.autoscale(True, axis='both', tight=True)
        plt.plot(xlabels, event.get_waveform('uS1').samples[lefti:righti + 1], label='S1 peakfinding')
        plt.plot(xlabels, event.get_waveform('uS2').samples[lefti:righti + 1], label='S2 peakfinding')
        plt.plot(xlabels, event.get_waveform('filtered_for_large_s2').samples[lefti:righti + 1], label='Large s2 filtered')
        plt.plot(xlabels, event.get_waveform('filtered_for_small_s2').samples[lefti:righti + 1], label='Small s2 filtered')
        plt.plot(xlabels, event.get_waveform('veto').samples[lefti:righti + 1], label='Veto')
        plt.plot(xlabels, [0.6241506363] * nsamples,  '--', label='Large S2 threshold')
        plt.plot(xlabels, [0.06241506363] * nsamples,  '--', label='Small S2 threshold')
        plt.plot(xlabels, [0.1872451909] * nsamples,  '--', label='S1 threshold')
        if show_peaks:
            # Plot all peaks
            max_y = max([p.height for _, p in event.get_all_peaks()])

            for peak_type, peak in event.get_all_peaks():
                x = peak.index_of_maximum * self.time_conversion_factor
                y = peak.height
                plt.hlines(y, peak.left * self.time_conversion_factor, peak.right * self.time_conversion_factor)
                plt.annotate('%s:%s' % (peak_type, int(peak.area)),
                             xy=(x, y),
                             xytext=(x, y + (max_y - y) * (0.05 + 0.2 * random.random())),
                             arrowprops=dict(arrowstyle="fancy",
                                             fc="0.6", ec="none",
                                             connectionstyle="angle3,angleA=0,angleB=-90"))

            # plt.plot(event.get_waveform('tpc').samples,  label='TPC')
            # plt.plot(event.get_waveform('filtered_for_s2').samples, label='TPC - filtered')

        """
        # Plot all peaks
        max_y = max([p.height for p in event.S2s + event.S1s])

        for peak_type, peak in event.get_all_peaks():

            x = peak.index_of_maximum
            y = peak.height

            plt.hlines(y, peak.left, peak.right)
            ax.annotate('%s:%s' % (peak_type, int(peak.area)),
                        xy=(x, y),
                        xytext=(x, y + (max_y-y)*(0.05+0.2*random.random())),
                        arrowprops=dict(arrowstyle="fancy",
                                        fc="0.6", ec="none",
                                        connectionstyle="angle3,angleA=0,angleB=-90"))

        # plt.plot(event.get_waveform('tpc').samples,  label='TPC')
        # plt.plot(event.get_waveform('filtered_for_s2').samples, label='TPC - filtered')



        plt.xlabel('Sample in event [10 ns]')
        plt.ylabel("pe / bin")
        plt.tight_layout()

        plt.show(block=False)
        self.log.info("Hit enter to continue...")
        input()
        plt.close()
        """


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
